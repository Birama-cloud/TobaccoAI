"""Éléments communs à toutes les pages : feuille de style et en-tête de navigation."""

from pathlib import Path

import streamlit as st

from ui.formatting import brand_html

CSS_FILE = Path(__file__).with_name("style.css")


def load_css():
    st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_header(pages, current):
    """Logo + nom de la plateforme à gauche, liens de navigation à droite."""
    if current.url_path == pages["espace"].url_path:
        _render_data_header(pages)
        return
    with st.container(key="app-header"):
        left, right = st.columns([2, 3], vertical_alignment="center")
        left.markdown(brand_html(), unsafe_allow_html=True)
        with right:
            with st.container(key="app-nav", horizontal=True, horizontal_alignment="right",
                              vertical_alignment="center", gap="small"):
                for slug, page in pages.items():
                    actif = "-actif" if page.url_path == current.url_path else ""
                    with st.container(key=f"nav-{slug}{actif}", width="content"):
                        st.page_link(page, label=page.title)


def _render_data_header(pages):
    """En-tête sombre de l'Espace données (maquette 3)."""
    with st.container(key="app-header-dark"):
        left, right = st.columns([3, 1], vertical_alignment="center")
        left.markdown(brand_html(badge="Espace données"), unsafe_allow_html=True)
        with right, st.container(key="back-home", horizontal=True, horizontal_alignment="right"):
            st.page_link(pages["accueil"], label="← Retour à l'accueil")
