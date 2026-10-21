import streamlit as st

from ui.formatting import coming_soon_html


def render():
    st.markdown(coming_soon_html(
        "Cartographie des structures",
        "Une carte des structures de dépistage des cancers, d'aide à l'arrêt du tabac et de prise en charge du "
        "paludisme, par région, établie à partir de sources officielles et exportable en CSV.",
    ), unsafe_allow_html=True)
