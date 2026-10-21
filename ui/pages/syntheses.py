import streamlit as st

from ui.formatting import coming_soon_html


def render():
    st.markdown(coming_soon_html(
        "Synthèses documentaires",
        "Pour chaque thème, une synthèse de l'état des connaissances au Sénégal, rédigée à partir des documents "
        "du corpus : données disponibles, recommandations internationales et lacunes, avec les références prêtes à citer.",
    ), unsafe_allow_html=True)
