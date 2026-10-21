"""
Indexe les documents de tous les thèmes (ou d'un seul) et recopie les
informations de documents/sources.json dans la base ChromaDB.

Usage :
    python index_corpus.py                    (documents nouveaux uniquement)
    python index_corpus.py --theme paludisme
    python index_corpus.py --force            (réindexe tout depuis zéro)
"""

import argparse
import sys

import config
from rag.sources import RegistryError, load_registry
from rag.store import get_chroma_client, get_collection, index_theme, load_embedder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexe le corpus documentaire.")
    parser.add_argument("--theme", choices=list(config.THEMES), help="Un seul thème (par défaut : tous)")
    parser.add_argument("--force", action="store_true", help="Réindexer depuis zéro")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        registry = load_registry()
    except RegistryError as e:
        print(f"❌ {e}")
        sys.exit(1)

    embedder, client = load_embedder(), get_chroma_client()
    for theme in [args.theme] if args.theme else list(config.THEMES):
        collection = get_collection(client, theme)
        added = index_theme(theme, embedder, collection, force_reindex=args.force, registry=registry)
        print(f"{config.THEMES[theme]['label']} : {collection.count()} morceaux ({added} ajoutés)")
    print("Redémarrez l'application Streamlit pour qu'elle prenne en compte les changements.")
