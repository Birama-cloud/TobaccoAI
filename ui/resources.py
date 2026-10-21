"""Ressources partagées par toutes les pages, chargées une seule fois (mise en cache)."""

import streamlit as st

import config
from rag.sources import RegistryError, load_registry
from rag.store import get_chroma_client, get_collection, index_theme, load_embedder


@st.cache_resource(show_spinner="Chargement du modèle d'embedding…")
def get_embedder():
    return load_embedder()


@st.cache_resource
def get_chroma():
    return get_chroma_client()


@st.cache_resource(show_spinner="Préparation du corpus documentaire (première ouverture)…")
def prepare_corpus():
    """
    Au démarrage du serveur : indexe les PDF pas encore indexés et recopie les
    informations de documents/sources.json dans la base. Renvoie les
    avertissements (PDF absent du registre, registre mal formé...).
    Après un ajout de document : redémarrer l'application ou lancer
    `python index_corpus.py`.
    """
    messages = []
    try:
        registry = load_registry()
    except RegistryError as e:
        messages.append(f"❌ Registre des sources illisible : {e}")
        registry = {}
    embedder, client = get_embedder(), get_chroma()
    for theme in config.THEMES:
        index_theme(theme, embedder, get_collection(client, theme), log=messages.append, registry=registry)
    return [m for m in messages if m.startswith(("⚠", "❌"))]


def get_settings():
    """Réglages du modèle (.env). Arrête la page avec un message clair s'ils sont invalides."""
    try:
        return config.load_settings()
    except ValueError as e:
        st.error(f"Configuration invalide (fichier .env) : {e}")
        st.stop()
