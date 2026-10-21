import os
import re
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import pandas as pd

# ==========================================
# Étape 1: CONFIGURATION DE LA PAGE & STYLE CSS
# ==========================================
st.set_page_config(
    page_title="Assistant GATS-2023 | Sénégal",
    page_icon="🇸🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}
.stButton>button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
    background-color: #007bff;
    color: white;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background-color: #0056b3;
    color: white;
}
.chat-bubble {
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 5px solid #007bff;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.source-tag {
    display: inline-block;
    background-color: #eef4ff;
    color: #0056b3;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.85rem;
    margin: 0.15rem 0.3rem 0.15rem 0;
    border: 1px solid #cfe0ff;
}
.no-answer-box {
    padding: 1rem;
    border-radius: 8px;
    background-color: #fff8e6;
    border-left: 5px solid #ffb300;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# PARAMÈTRES RAG (réglables)
# ==========================================
DOCS_DIR = "documents"           # dossier contenant tous les PDF à indexer
LEGACY_PDF_FILE = "GATS_2023.pdf"  # ancien emplacement, migré automatiquement si présent
CHUNK_SIZE_WORDS = 220          # taille de chunk en mots (par page, jamais à cheval sur 2 pages)
CHUNK_OVERLAP_WORDS = 40        # chevauchement entre chunks consécutifs
TOP_K = 5                        # nombre de chunks récupérés
# Distance cosinus max acceptée (0 = identique, 2 = opposé).
# 0.65 s'est révélé trop strict : une simple faute de frappe ("sengal" au lieu
# de "Sénégal") suffisait à rejeter une réponse par ailleurs pertinente.
# 0.78 laisse plus de marge pour les variations de formulation, tout en
# filtrant les questions réellement hors-sujet (distance typiquement >1.0).
# À recalibrer avec un vrai jeu de test (voir priorité 6 du plan).
RELEVANCE_DISTANCE_THRESHOLD = 0.78
CHROMA_PERSIST_DIR = "./chroma_db"

# ==========================================
# GESTION DE LA SESSION
# ==========================================
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'history' not in st.session_state:
    st.session_state.history = []

os.makedirs(DOCS_DIR, exist_ok=True)

# Migration automatique : si l'ancien GATS_2023.pdf est à la racine et que le
# dossier documents/ est vide, on le déplace pour ne pas casser les anciennes
# installations.
if os.path.exists(LEGACY_PDF_FILE):
    legacy_target = os.path.join(DOCS_DIR, LEGACY_PDF_FILE)
    if not os.path.exists(legacy_target):
        import shutil
        shutil.copy(LEGACY_PDF_FILE, legacy_target)


def list_pdf_files(docs_dir=DOCS_DIR):
    """Liste tous les PDF disponibles dans le dossier documents/."""
    if not os.path.isdir(docs_dir):
        return []
    return sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf"))


def get_indexed_sources(collection):
    """Renvoie {nom_de_fichier: nombre_de_chunks} pour tout ce qui est déjà indexé."""
    if collection.count() == 0:
        return {}
    data = collection.get(include=["metadatas"])
    counts = {}
    for meta in data["metadatas"]:
        src = meta.get("source", "inconnu")
        counts[src] = counts.get(src, 0) + 1
    return counts


# ==========================================
# MODÈLES (mis en cache) — chargés AVANT la sidebar, qui en a besoin
# ==========================================
@st.cache_resource
def get_models():
    # Modèle multilingue (au lieu de all-MiniLM-L6-v2, entraîné quasi
    # exclusivement en anglais). Essentiel pour un rapport en français :
    # "all-MiniLM-L6-v2" capte mal le lien sémantique entre une question
    # comme "les jeunes" et une formulation du rapport comme "15-24 ans".
    sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
    chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # cosine distance -> plus interprétable pour un seuil de pertinence que L2 brut
    # Nom de collection changé (vs "gats_rag") pour forcer une réindexation propre
    # avec le nouveau modèle multilingue : les anciens vecteurs ne sont pas compatibles.
    collection = chroma.get_or_create_collection(
        name="gats_rag_multilingual",
        metadata={"hnsw:space": "cosine"}
    )

    return sbert, collection


sbert_model, collection = get_models()

# Si la base persistante contient déjà des données d'une session précédente,
# on considère le document comme indexé sans repasser par le bouton.
if not st.session_state.indexed and collection.count() > 0:
    st.session_state.indexed = True


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    if os.path.exists("image/logo_ansd.webp"):
        st.image("image/logo_ansd.webp", width=150)
    st.title("Configuration")

    st.markdown("---")

    openai_api_key = st.text_input(
        "Clé API OpenAI",
        type="password",
        help="Entrez votre clé API OpenAI pour activer les réponses.",
        placeholder="sk-..."
    )

    if not openai_api_key:
        st.warning("⚠ Veuillez saisir votre clé API OpenAI pour continuer.")
    else:
        st.success("✅ Clé API détectée")
        st.markdown("---")
        st.subheader("📄 Base documentaire")

        uploaded_files = st.file_uploader(
            "Ajouter un ou plusieurs rapports PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help="Ajoute de nouveaux documents (rapports OMS, études, rapports nationaux...) sans toucher au code."
        )
        if uploaded_files:
            newly_saved = []
            for uf in uploaded_files:
                dest_path = os.path.join(DOCS_DIR, uf.name)
                if not os.path.exists(dest_path):
                    with open(dest_path, "wb") as f:
                        f.write(uf.getbuffer())
                    newly_saved.append(uf.name)
            if newly_saved:
                st.success(f"Ajouté(s) : {', '.join(newly_saved)}. Lance l'indexation ci-dessous.")

        available_files = list_pdf_files()
        indexed_sources = get_indexed_sources(collection)

        if not available_files:
            st.error(f"❌ Aucun PDF trouvé. Dépose un fichier ci-dessus ou place-le dans le dossier '{DOCS_DIR}/'.")
        else:
            st.caption(f"{len(available_files)} document(s) dans '{DOCS_DIR}/' :")
            for fname in available_files:
                if fname in indexed_sources:
                    st.caption(f"✅ {fname} — {indexed_sources[fname]} morceaux indexés")
                else:
                    st.caption(f"⏳ {fname} — pas encore indexé")

            if st.button("🚀 Indexer les nouveaux documents"):
                st.session_state.start_indexing = True

            if st.session_state.indexed:
                if st.button("🔄 Tout réindexer depuis zéro"):
                    st.session_state.force_reindex = True
                    st.session_state.start_indexing = True

        st.markdown("---")
        with st.expander("⚙️ Réglages avancés"):
            st.caption(f"Seuil de pertinence (distance cosinus) : **{RELEVANCE_DISTANCE_THRESHOLD}**")
            st.caption(f"Chunks récupérés par question : **{TOP_K}**")
            st.caption("Un seuil plus bas = le système répond 'non trouvé' plus souvent (plus prudent).")

        st.markdown("---")
        st.caption("Développé pour les étudiants en IA - Guide RAG")


# ==========================================
# Étape 3: Extraction du Texte du PDF — PAR PAGE
# ==========================================
def extract_pages_from_pdf(pdf_path):
    """Extrait le texte page par page pour pouvoir citer un numéro de page précis."""
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            content = page.extract_text()
            if content and content.strip():
                pages.append((page_number, content))
        return pages
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du PDF : {e}")
        return []


# ==========================================
# Étape 4: Chunking — jamais à cheval sur deux pages
# ==========================================
def split_into_sentences(text):
    """Découpage simple en phrases pour éviter de couper au milieu d'une idée."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_page_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """
    Découpe le texte d'UNE page en chunks avec chevauchement, en respectant
    autant que possible les frontières de phrases.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_words = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_words) + len(sentence_words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # chevauchement : on garde les derniers mots du chunk précédent
            current_words = current_words[-overlap:] if overlap else []
        current_words.extend(sentence_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ==========================================
# Étape 5: Indexation complète — multi-documents, avec métadonnées (source + page)
# ==========================================
def sanitize_id_prefix(filename):
    """Rend un nom de fichier utilisable comme préfixe d'ID ChromaDB."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', filename)


def process_documents(docs_dir=DOCS_DIR, force_reindex=False):
    """
    Pipeline complet, exécuté sur TOUS les PDF du dossier documents/ :
    Extraction -> Chunking par page -> Embedding -> Stockage avec métadonnées.
    Ne réindexe que les documents pas encore présents dans la collection,
    sauf si force_reindex=True (repart alors de zéro pour tout le monde).
    """
    with st.spinner("Extraction et indexation des documents en cours..."):

        if force_reindex and collection.count() > 0:
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)

        already_indexed = get_indexed_sources(collection)
        all_files = list_pdf_files(docs_dir)
        files_to_process = [f for f in all_files if f not in already_indexed]

        if not files_to_process:
            st.info("Tous les documents disponibles sont déjà indexés.")
            st.session_state.indexed = collection.count() > 0
            return True

        total_chunks_added = 0

        for fname in files_to_process:
            path = os.path.join(docs_dir, fname)
            pages = extract_pages_from_pdf(path)
            if not pages:
                st.warning(f"Aucun texte extrait de {fname} — ignoré.")
                continue

            id_prefix = sanitize_id_prefix(fname)
            doc_chunks, doc_metadatas, doc_ids = [], [], []
            chunk_counter = 0

            for page_number, page_text in pages:
                for chunk in chunk_page_text(page_text):
                    doc_chunks.append(chunk)
                    doc_metadatas.append({"source": fname, "page": page_number})
                    doc_ids.append(f"{id_prefix}_{chunk_counter}")
                    chunk_counter += 1

            if not doc_chunks:
                continue

            embeddings = sbert_model.encode(doc_chunks, show_progress_bar=False, normalize_embeddings=True)

            collection.add(
                documents=doc_chunks,
                embeddings=embeddings.tolist(),
                metadatas=doc_metadatas,
                ids=doc_ids
            )
            total_chunks_added += len(doc_chunks)
            st.success(f"{fname} indexé : {len(doc_chunks)} morceaux sur {len(pages)} pages.")

        if total_chunks_added == 0:
            st.warning("Aucun nouveau contenu n'a pu être indexé.")

        st.session_state.indexed = collection.count() > 0
        return True


if st.session_state.get('start_indexing'):
    if not openai_api_key:
        st.error("Saisissez d'abord votre clé API !")
    else:
        force = st.session_state.get('force_reindex', False)
        process_documents(DOCS_DIR, force_reindex=force)
    del st.session_state['start_indexing']
    st.session_state.pop('force_reindex', None)
    st.rerun()


# ==========================================
# Étape 6: Recherche (Retrieval) — avec métadonnées et scores
# ==========================================
def search_context(query, top_k=TOP_K):
    """
    Recherche les chunks les plus pertinents.
    Retourne une liste de dicts : {text, source, page, distance}
    """
    # Normalisation légère de la question avant l'embedding : les modèles
    # d'embedding sont entraînés sur du texte à la casse "naturelle". Une
    # question tapée TOUT EN MAJUSCULES ou avec une casse inhabituelle
    # produit un vecteur plus éloigné du texte indexé, ce qui peut faire
    # rater une information pourtant présente. On ne touche pas au texte
    # affiché à l'utilisateur, seulement à la version envoyée au modèle.
    normalized_query = " ".join(query.strip().lower().split())

    query_embedding = sbert_model.encode([normalized_query], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    hits = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        hits.append({
            "text": doc,
            "source": meta.get("source", "document"),
            "page": meta.get("page", "?"),
            "distance": dist
        })

    return hits


# ==========================================
# Étape 7: Génération avec OpenAI (LLM) — prompt avec pages
# ==========================================
def generate_answer(query, hits, openai_api_key):
    """Génère une réponse en citant explicitement document + page pour chaque passage utilisé."""

    context_blocks = []
    for i, h in enumerate(hits, start=1):
        context_blocks.append(f"[Extrait {i} — {h['source']}, page {h['page']}]\n{h['text']}")
    context_text = "\n\n".join(context_blocks)

    try:
        openai_client = OpenAI(api_key=openai_api_key)

        prompt = f"""Tu es un expert en santé publique spécialisé dans la lutte contre le tabagisme en Afrique.

CONSIGNE :
- Réponds à la question UNIQUEMENT en utilisant les extraits du CONTEXTE ci-dessous.
- Chaque extrait est étiqueté avec son document et son numéro de page.
- Il est NORMAL que le contexte contienne plusieurs chiffres proches sur des sous-catégories différentes
  (ex : prévalence globale, par sexe, par âge, fumeurs quotidiens vs occasionnels...). Ce n'est PAS une
  raison de refuser de répondre. Choisis le chiffre qui répond le plus directement à la question posée
  (par défaut, préfère la statistique "ensemble"/"global" si la question ne précise pas de sous-groupe),
  et mentionne brièvement s'il existe des variantes pertinentes (par sexe, âge, milieu...).
- Quand tu utilises une information, indique entre parenthèses le document et la page correspondante, par exemple (GATS_2023.pdf, page 12).
- Ne refuse de répondre QUE si la question porte sur un sujet réellement absent du contexte fourni
  (aucun chiffre, même approximatif, ne s'en approche). Dans ce cas seulement, dis clairement que tu ne
  trouves pas l'information dans les documents indexés. N'invente jamais de chiffre, de page ou de document.

CONTEXTE :
{context_text}

QUESTION :
{query}

RÉPONSE (avec document + page entre parenthèses) :
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un assistant expert basé sur des documents officiels. Tu ne réponds qu'à partir du contexte fourni et tu cites toujours la page."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Erreur lors de la génération de la réponse : {e}. Vérifiez votre clé API ou votre connexion."


# ==========================================================
# Étape 8: Tableau de bord — indicateurs clés (données vérifiées manuellement
# dans le rapport GATS Sénégal 2023, avec page source pour chaque chiffre).
# Ce n'est PAS généré par le LLM : ce sont des chiffres fixes et sourcés,
# donc aucun risque d'hallucination sur cette partie de l'application.
# ===========================================================

# ===========================================================
# Étape 8: Tableau de bord — indicateurs clés (GATS Sénégal 2023)
# ===========================================================

import pandas as pd
import streamlit as st

DASHBOARD_SOURCE_LABEL = "GATS Sénégal 2023 (ANSD)"


def render_dashboard():
    # Style CSS personnalisé pour un look moderne et épuré
    st.markdown(
        """
        <style>
            /* Style des cartes pour les graphiques */
            .stChartBlock {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            }
            /* En-tête élégant */
            .header-badge {
                background-color: #e0f2fe;
                color: #0369a1;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
                display: inline-block;
                margin-bottom: 8px;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Header / Titre principal
    st.markdown(
        '<span class="header-badge">Données Officielle ANSD</span>',
        unsafe_allow_html=True,
    )
    st.title("📊 Indicateurs Clés — Tabagisme au Sénégal")
    st.caption(
        f"Chiffres vérifiés manuellement dans le rapport **{DASHBOARD_SOURCE_LABEL}**. Contenu fixe, indépendant de la base indexée."
    )

    st.write("")  # Espace

    # ---- KPIs principaux (Bordures modernes) ----
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        with st.container(border=True):
            st.metric(
                label="Prévalence globale",
                value="4,4 %",
                delta="-1,6 pt vs 2015",
                delta_color="inverse",
            )

    with kpi_col2:
        with st.container(border=True):
            st.metric(
                label="Connaissance C-Électronique",
                value="36,6 %",
                help="Usage actuel : seulement 0,4 % (p. 15, 94)",
            )

    with kpi_col3:
        with st.container(border=True):
            st.metric(
                label="Intention d'arrêter",
                value="62,5 %",
                delta="-17,3 pt vs 2015",
                delta_color="normal",
            )

    with kpi_col4:
        with st.container(border=True):
            st.metric(
                label="Prix moyen d'un paquet",
                value="849 FCFA",
                delta="+21,2 % vs 2015",
                delta_color="normal",
            )

    st.caption("📌 *Sources KPIs : rapport GATS 2023, page 94.*")

    st.divider()

    # ---- Graphiques : Section 1 ----
    chart_col1, chart_col2 = st.columns(2, gap="medium")

    with chart_col1:
        with st.container(border=True):
            st.subheader("Prévalence par tranche d'âge")
            df_age = pd.DataFrame(
                {"Prévalence (%)": [1.6, 6.2, 6.3, 3.0]},
                index=["15-24 ans", "25-44 ans", "45-64 ans", "65 ans et +"],
            )
            st.bar_chart(df_age, color="#1f77b4")
            st.caption("Source : GATS 2023, page 49-50.")

    with chart_col2:
        with st.container(border=True):
            st.subheader("Prévalence par sexe")
            df_sexe = pd.DataFrame(
                {"Prévalence (%)": [8.2, 0.8]}, index=["Hommes", "Femmes"]
            )
            st.bar_chart(df_sexe, color="#2ca02c")
            st.caption("Source : GATS 2023, page 50 (tableau 4.10).")

    st.write("")

    # ---- Graphiques : Section 2 ----
    chart_col3, chart_col4 = st.columns(2, gap="medium")

    with chart_col3:
        with st.container(border=True):
            st.subheader("Prévalence par milieu de résidence")
            df_milieu = pd.DataFrame(
                {"Prévalence (%)": [5.0, 3.9]}, index=["Urbain", "Rural"]
            )
            st.bar_chart(df_milieu, color="#ff7f0e")
            st.caption("Source : GATS 2023, page 49.")

    with chart_col4:
        with st.container(border=True):
            st.subheader("Statut des fumeurs (adulte)")
            df_statut = pd.DataFrame(
                {"Part des adultes (%)": [95.9, 3.4, 0.7]},
                index=[
                    "Non-fumeurs",
                    "Fumeurs quotidiens",
                    "Fumeurs occasionnels",
                ],
            )
            st.bar_chart(df_statut, color="#d62728")
            st.caption("Source : GATS 2023, page 37 (graphique 4.1).")

    st.divider()

    # ---- Graphique : Évolution ----
    with st.container(border=True):
        st.subheader("📈 Évolution comparative (2015 → 2023)")
        df_trend = pd.DataFrame(
            {
                "2015": [6.0, 79.8, 59.6],
                "2023": [4.4, 62.5, 52.4],
            },
            index=[
                "Prévalence globale (%)",
                "Intention d'arrêter (%)",
                "Tentative d'arrêt (%)",
            ],
        )
        st.bar_chart(df_trend)
        st.caption("Source : GATS 2023, page 94 (synthèse des tendances).")

# ==========================================================
# Étape 9: Interface principale
# ===========================================================
def render_chat_tab():
    if not st.session_state.indexed:
        st.info("👋 Bienvenue ! Saisissez votre clé OpenAI dans la barre latérale, déposez au moins un rapport PDF, puis cliquez sur '🚀 Indexer les nouveaux documents'.")
        return

    indexed_sources = get_indexed_sources(collection)
    st.caption(f"📚 Base actuelle : {len(indexed_sources)} document(s) indexé(s) — {sum(indexed_sources.values())} extraits au total.")

    query = st.chat_input("Posez votre question sur les rapports indexés (ex: Quel est le taux de tabagisme chez les jeunes ?)")

    if not query:
        return

    if not openai_api_key:
        st.error("Veuillez saisir votre clé API OpenAI dans la barre latérale pour obtenir une réponse.")
        return

    with st.spinner("🔍 Recherche dans le rapport et génération de la réponse..."):

        # 1. Retrieval
        hits = search_context(query)

        # 2. Garde-fou anti-hallucination : si même le meilleur résultat est trop
        #    éloigné sémantiquement, on ne génère pas de réponse inventée.
        best_distance = hits[0]["distance"] if hits else 1.0
        relevant_hits = [h for h in hits if h["distance"] <= RELEVANCE_DISTANCE_THRESHOLD]

        if not relevant_hits:
            st.markdown(f"""
            <div class="chat-bubble">
                <strong>Question :</strong> {query}
            </div>
            <div class="no-answer-box">
                🤔 Je ne trouve pas d'information suffisamment pertinente dans les documents actuellement indexés pour répondre à cette question avec certitude.
                Essayez de reformuler, ou vérifiez que le bon document a été ajouté et indexé.
            </div>
            """, unsafe_allow_html=True)
            with st.expander("🔧 Détail technique (score de pertinence le plus proche trouvé)"):
                st.caption(f"Distance la plus faible obtenue : {best_distance:.3f} (seuil accepté : {RELEVANCE_DISTANCE_THRESHOLD})")
            return

        # 3. Generation
        answer = generate_answer(query, relevant_hits, openai_api_key)

        # 4. Affichage réponse
        st.markdown(f"""
        <div class="chat-bubble">
            <strong>Question :</strong> {query}<br><br>
            <strong>Réponse de l'expert :</strong><br>
            {answer}
        </div>
        """, unsafe_allow_html=True)

        # 5. Tags de sources (document + page), déduplicés
        seen_pages = set()
        tags_html = ""
        for h in relevant_hits:
            key = (h["source"], h["page"])
            if key not in seen_pages:
                seen_pages.add(key)
                tags_html += f'<span class="source-tag">📄 {h["source"]} — page {h["page"]}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)

        # 6. Extraits détaillés
        with st.expander("📚 Voir les extraits utilisés"):
            for i, h in enumerate(relevant_hits, start=1):
                st.markdown(f"**Extrait {i} — {h['source']}, page {h['page']}** (pertinence : {1 - h['distance']:.2f})")
                st.caption(h["text"])
                st.markdown("---")


def main():
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("image/drapeau.webp"):
            st.image("image/drapeau.webp", width=200)
    with col2:
        if os.path.exists("image/image_tabac.webp"):
            st.image("image/image_tabac.webp", width=400)

    st.title(" Assistant Intelligent Tabagisme — Sénégal & au-delà")
    st.subheader("Interrogez plusieurs rapports (GATS, OMS, études...) avec l'IA")

    tab_chat, tab_dashboard = st.tabs(["💬 Assistant", "📊 Tableau de bord"])

    with tab_chat:
        render_chat_tab()

    with tab_dashboard:
        render_dashboard()

    st.markdown("---")
    st.markdown(
        "<center><small>Guide pédagogique RAG - Base documentaire tabagisme (GATS 2023 et autres sources)</small></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()