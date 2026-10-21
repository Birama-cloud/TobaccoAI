"""
Point d'entrée de l'application Streamlit.
Lancement :  streamlit run application.py
"""

import streamlit as st

import config
from ui.layout import load_css, render_header
from ui.navigation import build_pages

st.set_page_config(
    page_title=config.NOM_PLATEFORME,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css()
pages = build_pages()
current = st.navigation(list(pages.values()), position="hidden")
render_header(pages, current)
current.run()
