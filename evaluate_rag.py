"""
Évaluation technique du RAG — TobaccoAI
=========================================
Priorité 6 du plan de transformation : mesurer objectivement la fiabilité
du pipeline RAG avant de le présenter au concours, au lieu d'avancer des
résultats non mesurés.

Ce script :
  1. Se connecte à la même base vectorielle persistante que l'application
     (./chroma_db, collection "gats_rag_multilingual") — il teste donc
     exactement ce qui est réellement indexé, pas une copie recréée.
  2. Pour chaque question du jeu de test (eval_questions.json), effectue
     la même recherche + génération que l'app, et vérifie :
       - si la page attendue a été retrouvée (pertinence du retrieval)
       - si les mots-clés attendus apparaissent dans la réponse (exactitude)
       - si le garde-fou anti-hallucination réagit correctement sur les
         questions "pièges" (hors-sujet ou information absente du rapport)
       - le temps de réponse
  3. Affiche un rapport résumé et sauvegarde le détail dans un CSV.

Usage :
    pip install sentence-transformers chromadb openai --break-system-packages
    python evaluate_rag.py --api-key sk-...
    (ou définir la variable d'environnement OPENAI_API_KEY)

IMPORTANT : lancez d'abord l'application Streamlit et indexez vos documents
avant d'exécuter ce script — il ne réindexe rien lui-même.
"""

import argparse
import csv
import json
import os
import sys
import time

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# ⚠️ Ces constantes doivent rester identiques à celles de application.py.
# Si tu changes le modèle d'embedding, le seuil, ou le nombre de chunks
# récupérés dans application.py, reporte le changement ici aussi.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "gats_rag_multilingual"
TOP_K = 5
RELEVANCE_DISTANCE_THRESHOLD = 0.78

REFUSAL_MARKERS = [
    "je ne trouve pas",
    "ne trouve pas d'information",
    "pas d'information",
    "n'est pas présente",
    "non trouvé",
]


def load_test_set(path="eval_questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_context(sbert_model, collection, query, top_k=TOP_K):
    normalized_query = " ".join(query.strip().lower().split())
    query_embedding = sbert_model.encode([normalized_query], normalize_embeddings=True)[0].tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({
            "text": doc,
            "source": meta.get("source", "document"),
            "page": meta.get("page", "?"),
            "distance": dist
        })
    return hits


def generate_answer(client, query, hits):
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        context_blocks.append(f"[Extrait {i} — {h['source']}, page {h['page']}]\n{h['text']}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"""Tu es un expert en santé publique spécialisé dans la lutte contre le tabagisme en Afrique.

CONSIGNE :
- Réponds à la question UNIQUEMENT en utilisant les extraits du CONTEXTE ci-dessous.
- Chaque extrait est étiqueté avec son document et son numéro de page.
- Quand tu utilises une information, indique entre parenthèses le document et la page correspondante, par exemple (GATS_2023.pdf, page 12).
- Si l'information n'est présente dans AUCUN extrait, dis clairement que tu ne la trouves pas dans les documents indexés. N'invente jamais de chiffre, de page ou de document.

CONTEXTE :
{context_text}

QUESTION :
{query}

RÉPONSE (avec document + page entre parenthèses) :
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un assistant expert basé sur des documents officiels. Tu ne réponds qu'à partir du contexte fourni et tu cites toujours la page."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


def looks_like_refusal(answer_text):
    lowered = answer_text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def run_evaluation(api_key, test_set_path="eval_questions.json", output_csv="eval_results.csv"):
    print("Chargement du modèle d'embedding...")
    sbert_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connexion à la base vectorielle persistante...")
    chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    if collection.count() == 0:
        print("❌ La base vectorielle est vide. Lance d'abord l'app Streamlit et indexe tes documents.")
        sys.exit(1)

    print(f"Base connectée : {collection.count()} chunks indexés.\n")

    client = OpenAI(api_key=api_key)
    test_set = load_test_set(test_set_path)

    results = []

    for item in test_set:
        qid = item["id"]
        question = item["question"]
        expects_answer = item["expects_answer"]
        expected_keywords = item.get("expected_keywords", [])
        expected_pages = set(item.get("expected_pages", []))

        print(f"[{qid}] {question}")
        start = time.time()

        hits = search_context(sbert_model, collection, question)
        best_distance = hits[0]["distance"] if hits else 999
        relevant_hits = [h for h in hits if h["distance"] <= RELEVANCE_DISTANCE_THRESHOLD]

        retrieved_pages = {h["page"] for h in hits}
        expected_page_retrieved = bool(expected_pages & retrieved_pages) if expected_pages else None

        if relevant_hits:
            answer = generate_answer(client, question, relevant_hits)
        else:
            answer = "[Garde-fou] Aucun extrait suffisamment pertinent — réponse non générée."

        elapsed = time.time() - start

        system_refused = (not relevant_hits) or looks_like_refusal(answer)
        keyword_found = any(kw.lower() in answer.lower() for kw in expected_keywords) if expected_keywords else None

        if expects_answer:
            correct = bool(keyword_found) and not system_refused
        else:
            correct = system_refused  # bonne conduite = avoir refusé de répondre

        results.append({
            "id": qid,
            "question": question,
            "expects_answer": expects_answer,
            "expected_page_retrieved": expected_page_retrieved,
            "best_distance": round(best_distance, 3),
            "system_refused": system_refused,
            "keyword_found": keyword_found,
            "correct": correct,
            "response_time_sec": round(elapsed, 2),
            "answer": answer.replace("\n", " ")[:300],
        })

        verdict = "✅" if correct else "❌"
        print(f"    {verdict} distance={best_distance:.3f} | temps={elapsed:.1f}s | refus_systeme={system_refused}")
        print(f"    Réponse : {answer[:150]}...\n")

    # ---- Rapport résumé ----
    total = len(results)
    positives = [r for r in results if r["expects_answer"]]
    negatives = [r for r in results if not r["expects_answer"]]

    retrieval_hits = [r for r in positives if r["expected_page_retrieved"] is not None]
    retrieval_rate = (sum(r["expected_page_retrieved"] for r in retrieval_hits) / len(retrieval_hits) * 100) if retrieval_hits else 0

    answer_correct_rate = (sum(r["correct"] for r in positives) / len(positives) * 100) if positives else 0
    refusal_correct_rate = (sum(r["correct"] for r in negatives) / len(negatives) * 100) if negatives else 0
    overall_correct_rate = sum(r["correct"] for r in results) / total * 100
    avg_time = sum(r["response_time_sec"] for r in results) / total

    print("=" * 60)
    print("RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 60)
    print(f"Questions testées                         : {total}")
    print(f"  dont questions avec réponse attendue     : {len(positives)}")
    print(f"  dont questions pièges (pas de réponse)   : {len(negatives)}")
    print("-" * 60)
    print(f"Taux de récupération de la bonne page      : {retrieval_rate:.1f}%")
    print(f"Taux d'exactitude sur questions valides     : {answer_correct_rate:.1f}%")
    print(f"Taux de bon refus (anti-hallucination)      : {refusal_correct_rate:.1f}%")
    print(f"Taux de réussite global                     : {overall_correct_rate:.1f}%")
    print(f"Temps de réponse moyen                      : {avg_time:.2f}s")
    print("=" * 60)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDétail complet sauvegardé dans : {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue la fiabilité du pipeline RAG TobaccoAI.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="Clé API OpenAI (ou variable d'env OPENAI_API_KEY)")
    parser.add_argument("--test-set", default="eval_questions.json")
    parser.add_argument("--output", default="eval_results.csv")
    args = parser.parse_args()

    if not args.api_key:
        print("❌ Fournis une clé API avec --api-key sk-... ou via la variable d'environnement OPENAI_API_KEY.")
        sys.exit(1)

    run_evaluation(args.api_key, args.test_set, args.output)
