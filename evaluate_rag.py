"""
Évaluation du pipeline RAG, thème par thème
=============================================
Ce script appelle EXACTEMENT le même code que l'application
(rag.pipeline.answer_question : même recherche, même seuil, même prompt,
même modèle, tous réglés dans config.py). Il mesure donc ce que voit
réellement l'utilisateur.

Pour chaque question du jeu de test du thème (ex : eval_questions_tabac.json),
il vérifie :
  - si la page attendue a été retrouvée (pertinence de la recherche) ;
  - si les mots-clés attendus apparaissent dans la réponse (exactitude) ;
  - si le garde-fou réagit correctement sur les questions pièges ;
  - le temps de réponse.
Si la base du thème est vide, il indexe d'abord les documents du thème.

Usage :
    python evaluate_rag.py                     (thème tabac, clé lue dans .env)
    python evaluate_rag.py --theme tabac --api-key sk-...
    python evaluate_rag.py --theme tabac --pause 13   (pause entre les questions)
"""

import argparse
import csv
import json
import sys
import time

import config
from llm import create_llm_client
from rag.pipeline import answer_question
from rag.store import get_chroma_client, get_collection, index_theme, load_embedder


def load_test_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def page_retrieved(item, hits):
    """La page attendue (et le document attendu, s'il est précisé) figure-t-elle parmi les résultats ?"""
    expected_pages = set(item.get("expected_pages", []))
    if not expected_pages:
        return None
    expected_source = item.get("expected_source")
    return any(
        h["page"] in expected_pages and (expected_source is None or h["source"] == expected_source)
        for h in hits
    )


def evaluate_item(item, embedder, collections, llm):
    start = time.time()
    result = answer_question(item["question"], embedder, collections, llm)
    elapsed = time.time() - start

    if result.guardrail_triggered:
        answer = "[Garde-fou] Aucun extrait suffisamment pertinent — réponse non générée."
    elif result.error:
        answer = f"[Erreur] {result.error}"
    else:
        answer = result.answer

    expected_keywords = item.get("expected_keywords", [])
    keyword_found = any(kw.lower() in answer.lower() for kw in expected_keywords) if expected_keywords else None

    if result.error:
        correct = False
    elif item["expects_answer"]:
        correct = bool(keyword_found) and not result.refused
    else:
        correct = result.refused  # bonne conduite = avoir refusé de répondre

    return {
        "id": item["id"],
        "question": item["question"],
        "expects_answer": item["expects_answer"],
        "expected_page_retrieved": page_retrieved(item, result.hits),
        "best_distance": round(result.best_distance, 3) if result.best_distance is not None else None,
        "system_refused": result.refused,
        "keyword_found": keyword_found,
        "correct": correct,
        "error": result.error or "",
        "response_time_sec": round(elapsed, 2),
        "answer": answer.replace("\n", " ")[:300],
    }


def print_summary(theme, results):
    total = len(results)
    positives = [r for r in results if r["expects_answer"]]
    negatives = [r for r in results if not r["expects_answer"]]
    with_pages = [r for r in positives if r["expected_page_retrieved"] is not None]

    def rate(rows, key):
        return sum(bool(r[key]) for r in rows) / len(rows) * 100 if rows else 0

    print("=" * 60)
    print(f"RÉSUMÉ DE L'ÉVALUATION — thème : {config.THEMES[theme]['label']}")
    print("=" * 60)
    print(f"Questions testées                         : {total}")
    print(f"  dont questions avec réponse attendue     : {len(positives)}")
    print(f"  dont questions pièges (pas de réponse)   : {len(negatives)}")
    print("-" * 60)
    print(f"Taux de récupération de la bonne page      : {rate(with_pages, 'expected_page_retrieved'):.1f}%")
    print(f"Taux d'exactitude sur questions valides    : {rate(positives, 'correct'):.1f}%")
    print(f"Taux de bon refus (anti-hallucination)     : {rate(negatives, 'correct'):.1f}%")
    print(f"Taux de réussite global                    : {rate(results, 'correct'):.1f}%")
    print(f"Temps de réponse moyen                     : {sum(r['response_time_sec'] for r in results) / total:.2f}s")
    errors = sum(1 for r in results if r["error"])
    if errors:
        print(f"⚠ Erreurs d'appel au modèle              : {errors} (voir la colonne 'error' du CSV)")
    print("=" * 60)


def run_evaluation(theme, api_key=None, test_set_path=None, output_csv=None, pause=0):
    settings = config.load_settings()
    test_set_path = test_set_path or config.PROJECT_ROOT / config.THEMES[theme]["eval_file"]
    output_csv = output_csv or config.PROJECT_ROOT / f"eval_results_{theme}.csv"

    print(f"Thème évalué : {config.THEMES[theme]['label']} ({theme})")
    print(f"Modèle       : {settings.provider_label} — {settings.llm_model}")
    print(f"Jeu de test  : {test_set_path}\n")

    try:
        test_set = load_test_set(test_set_path)
    except FileNotFoundError:
        print(f"❌ Jeu de test introuvable : {test_set_path}")
        sys.exit(1)

    try:
        llm = create_llm_client(settings, api_key=api_key)
    except ValueError as e:
        print(f"❌ {e} Fournissez-la avec --api-key ou dans le fichier .env.")
        sys.exit(1)

    print("Chargement du modèle d'embedding...")
    embedder = load_embedder()
    collection = get_collection(get_chroma_client(), theme)

    if collection.count() == 0:
        print("Base vide pour ce thème : indexation des documents...")
        index_theme(theme, embedder, collection)
        if collection.count() == 0:
            print(f"❌ Aucun document indexable dans {config.theme_docs_dir(theme)}.")
            sys.exit(1)

    print(f"Base connectée : {collection.count()} morceaux indexés.\n")
    collections = {theme: collection}

    results = []
    for index, item in enumerate(test_set):
        if pause and index > 0:
            time.sleep(pause)  # reste sous la limite de requêtes par minute du fournisseur
        print(f"[{item['id']}] {item['question']}")
        row = evaluate_item(item, embedder, collections, llm)
        results.append(row)
        verdict = "✅" if row["correct"] else "❌"
        print(f"    {verdict} distance={row['best_distance']} | temps={row['response_time_sec']}s | refus={row['system_refused']}")
        print(f"    Réponse : {row['answer'][:150]}...\n")

    print_summary(theme, results)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDétail complet sauvegardé dans : {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue la fiabilité du pipeline RAG, thème par thème.")
    parser.add_argument("--theme", default="tabac", choices=list(config.THEMES))
    parser.add_argument("--api-key", default=None, help="Clé d'API (sinon lue dans .env)")
    parser.add_argument("--test-set", default=None, help="Jeu de questions (par défaut : celui du thème)")
    parser.add_argument("--output", default=None, help="Fichier CSV de sortie (par défaut : eval_results_<thème>.csv)")
    parser.add_argument("--pause", type=float, default=0,
                        help="Secondes d'attente entre deux questions (ex : 13 pour l'offre gratuite de Gemini, 5 requêtes/min)")
    args = parser.parse_args()

    # Évite une erreur d'affichage des accents/emojis dans certaines consoles Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run_evaluation(args.theme, args.api_key, args.test_set, args.output, args.pause)
