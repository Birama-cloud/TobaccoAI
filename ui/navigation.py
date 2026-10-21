"""Les cinq pages, dans l'ordre de la navigation."""

import streamlit as st

from ui.pages import accueil, assistant, cartographie, espace_donnees, syntheses

PAGES = {}


def build_pages():
    """Crée les pages (à chaque exécution) et les garde dans PAGES pour les liens internes."""
    PAGES.clear()
    PAGES.update({
        "accueil": st.Page(accueil.render, title="Accueil", url_path="accueil", default=True),
        "assistant": st.Page(assistant.render, title="Assistant", url_path="assistant"),
        "syntheses": st.Page(syntheses.render, title="Synthèses", url_path="syntheses"),
        "cartographie": st.Page(cartographie.render, title="Cartographie", url_path="cartographie"),
        "espace": st.Page(espace_donnees.render, title="Espace données", url_path="espace-donnees"),
    })
    return PAGES
