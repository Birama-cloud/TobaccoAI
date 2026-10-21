"""
Configuration centrale de la plateforme.

Tous les réglages (thèmes, modèle de langage, seuil de pertinence, TOP_K,
taille des morceaux...) sont ici. L'application Streamlit ET le script
d'évaluation lisent ce même fichier : ils testent donc exactement la même chose.

Les valeurs marquées "(.env)" peuvent être remplacées sans toucher au code,
dans un fichier .env à la racine du projet (voir .env.example).
"""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Nom affiché dans l'en-tête, en attendant le nom définitif.
NOM_PLATEFORME = "[Nom de la plateforme]"
DEPOT_GITHUB = "github.com/Birama-cloud/TobaccoAI"

# ==========================================
# THÈMES : un dossier de documents + une collection ChromaDB par thème
# ==========================================
DOCS_DIR = PROJECT_ROOT / "documents"

THEMES = {
    "tabac": {
        "label": "Tabac",
        "label_court": "Tabac",
        "collection": "theme_tabac",
        "sources_prevues": "GATS Sénégal, rapports OMS",
        "eval_file": "eval_questions_tabac.json",
        "couleur": "#9A3412",
        "fond": "#FFEDD5",
    },
    "cancer_sein": {
        "label": "Cancer du sein",
        "label_court": "Cancer du sein",
        "collection": "theme_cancer_sein",
        "sources_prevues": "Globocan (CIRC), OMS, ministère de la Santé",
        "eval_file": "eval_questions_cancer_sein.json",
        "couleur": "#9D174D",
        "fond": "#FCE7F3",
    },
    "cancer_col": {
        "label": "Cancer du col de l'utérus",
        "label_court": "Cancer du col",
        "collection": "theme_cancer_col",
        "sources_prevues": "Globocan (CIRC), recommandations OMS",
        "eval_file": "eval_questions_cancer_col.json",
        "couleur": "#0F766E",
        "fond": "#CCFBF1",
    },
    "paludisme": {
        "label": "Paludisme",
        "label_court": "Paludisme",
        "collection": "theme_paludisme",
        "sources_prevues": "PNLP, Rapport mondial sur le paludisme de l'OMS, EDS",
        "eval_file": "eval_questions_paludisme.json",
        "couleur": "#4338CA",
        "fond": "#E0E7FF",
    },
}
ALL_THEMES_LABEL = "Tous les thèmes"

# ==========================================
# REGISTRE DES SOURCES (documents/sources.json)
# ==========================================
SOURCES_REGISTRY = DOCS_DIR / "sources.json"

# Types de source : clé enregistrée dans ChromaDB -> libellé affiché (cases à cocher).
SOURCE_TYPES = {
    "enquete_nationale": "Enquêtes nationales (ANSD)",
    "oms": "Rapports et recommandations OMS",
    "circ": "Données CIRC (Globocan)",
    "pnlp": "Rapports du PNLP (paludisme)",
    "ministere": "Documents du ministère",
}
# Valeur enregistrée pour un PDF absent du registre (il reste cherchable sans filtre).
UNREGISTERED_SOURCE_TYPE = "non_repertorie"

# Questions suggérées dans l'Assistant. Uniquement des sujets traités par les
# documents indexés du thème : un thème sans document n'a pas de suggestion.
SUGGESTIONS = {
    "tabac": [
        "Comparer la prévalence du tabac chez les adultes au Sénégal en 2015 et en 2023",
        "Quelle est la prévalence du tabagisme chez les 15-24 ans au Sénégal ?",
        "Quelles sont les projections de l'OMS sur la consommation de tabac d'ici 2030 ?",
    ],
    "cancer_sein": [],
    "cancer_col": [],
    "paludisme": [],
}


def theme_docs_dir(theme):
    """Dossier contenant les PDF d'un thème (ex : documents/tabac)."""
    if theme not in THEMES:
        raise ValueError(f"Thème inconnu : {theme!r}. Thèmes possibles : {', '.join(THEMES)}")
    return DOCS_DIR / theme


# ==========================================
# RAG : découpage, embeddings, recherche
# ==========================================
CHUNK_SIZE_WORDS = 220      # taille d'un morceau en mots (jamais à cheval sur 2 pages)
CHUNK_OVERLAP_WORDS = 40    # chevauchement entre morceaux consécutifs
TOP_K = 5                   # nombre de morceaux récupérés par question

# Distance cosinus max acceptée (0 = identique, 2 = opposé).
# 0.65 s'est révélé trop strict : une simple faute de frappe ("sengal" au lieu
# de "Sénégal") suffisait à rejeter une réponse par ailleurs pertinente.
# 0.78 laisse plus de marge pour les variations de formulation, tout en
# filtrant les questions réellement hors-sujet (distance typiquement >1.0).
# À recalibrer avec le script d'évaluation.
RELEVANCE_DISTANCE_THRESHOLD = 0.78

# Modèle multilingue local (sentence-transformers) : adapté aux rapports en
# français, contrairement à all-MiniLM-L6-v2 entraîné surtout en anglais.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

# ==========================================
# MODÈLE DE LANGAGE (interchangeable)
# ==========================================
# Fournisseur par défaut (.env : LLM_PROVIDER). Les trois passent par l'API
# compatible OpenAI : seuls l'adresse du service et le nom du modèle changent.
DEFAULT_LLM_PROVIDER = "openai"
LLM_TEMPERATURE = 0

LLM_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-4o-mini",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "label": "Google Gemini",
        "default_model": None,  # à préciser dans .env (LLM_MODEL)
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        "label": "Ollama (modèle local)",
        "default_model": None,  # à préciser dans .env (LLM_MODEL), ex : le nom d'un modèle installé
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,    # pas de clé : le modèle tourne sur votre machine
    },
}

# ==========================================
# SANTÉ ET DONNÉES PERSONNELLES
# ==========================================
IMPORTANT_NOTICE = (
    "Les réponses sont générées à partir des documents indexés et peuvent contenir des erreurs : "
    "vérifiez les extraits cités. Vos questions ne sont pas enregistrées. "
    "N'y incluez aucune donnée permettant d'identifier une personne (nom, numéro, adresse, dossier médical)."
)

# Formulations qui signalent que le modèle n'a pas trouvé l'information.
REFUSAL_MARKERS = [
    "je ne trouve pas",
    "ne trouve pas d'information",
    "pas d'information",
    "n'est pas présente",
    "non trouvé",
]


# ==========================================
# Chargement des réglages modifiables (.env / variables d'environnement)
# ==========================================
@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_base_url: str | None
    llm_api_key: str | None   # clé lue dans .env (peut être vide : saisie dans l'interface)

    @property
    def needs_api_key(self):
        return LLM_PROVIDERS[self.llm_provider]["api_key_env"] is not None

    @property
    def provider_label(self):
        return LLM_PROVIDERS[self.llm_provider]["label"]


def load_settings(env=None, dotenv_path=PROJECT_ROOT / ".env"):
    """
    Lit les réglages modifiables. Si `env` n'est pas fourni, charge d'abord le
    fichier .env (sans écraser les variables déjà définies), puis os.environ.
    """
    if env is None:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path, override=False)
        env = os.environ

    provider = (env.get("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER={provider!r} inconnu. Valeurs possibles : {', '.join(LLM_PROVIDERS)}"
        )
    spec = LLM_PROVIDERS[provider]

    model = (env.get("LLM_MODEL") or "").strip() or spec["default_model"]
    if not model:
        raise ValueError(
            f"Indiquez le nom du modèle à utiliser avec {spec['label']} dans .env (LLM_MODEL=...)."
        )

    base_url = (env.get("LLM_BASE_URL") or "").strip() or spec["base_url"]
    api_key = None
    if spec["api_key_env"]:
        api_key = (env.get(spec["api_key_env"]) or "").strip() or None

    return Settings(llm_provider=provider, llm_model=model, llm_base_url=base_url, llm_api_key=api_key)
