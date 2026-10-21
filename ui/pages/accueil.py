"""
Page Accueil : accroche, recherche dans le corpus, chiffres clés, publics et thèmes.
Seuls les chiffres présents dans indicateurs.py sont affichés : les thèmes sans
document indiquent « données à venir ».
"""

import streamlit as st

import config
from indicateurs import format_number, get_value
from rag.store import get_collections, indexed_documents
from ui.formatting import escape_text
from ui.resources import get_chroma, prepare_corpus

PUBLICS = [
    ("Chercheurs", "Retrouver vite un chiffre, une méthode ou une recommandation, avec la référence exacte.",
     ["Réponses citées (document, page)", "Indicateurs téléchargeables en CSV", "Extraits consultables"],
     "assistant", "Interroger le corpus →"),
    ("Étudiants", "Comprendre un sujet de santé publique à partir des sources officielles, pour un mémoire ou un exposé.",
     ["Synthèses documentaires", "Références prêtes à citer", "Glossaire des indicateurs"],
     "syntheses", "Lire une synthèse →"),
    ("Organisations", "Préparer un plaidoyer, un projet ou un rapport avec des données à jour et une vue de l'offre de soins.",
     ["Tableaux de bord par thème", "Cartographie des structures", "Exports pour vos rapports"],
     "espace", "Ouvrir l'espace données →"),
]


def _kpi_card_html(theme):
    t = config.THEMES[theme]
    head = f'<div class="kpi-theme">{escape_text(t["label_court"].upper())}</div>'
    now, before = get_value(theme, "Prévalence globale", 2023), get_value(theme, "Prévalence globale", 2015)
    if now:
        evolution = f" ({format_number(before['valeur'])} % en 2015)" if before else ""
        body = (
            f'<div class="kpi-value">{format_number(now["valeur"])} %</div>'
            f'<div class="kpi-label">des adultes consomment du tabac{evolution}</div>'
            f'<div class="kpi-source">{escape_text(now["source"])}, page {escape_text(now["page"])}</div>'
        )
    else:
        body = (
            '<div class="kpi-value kpi-pending">Données à venir</div>'
            '<div class="kpi-label">Les indicateurs seront publiés quand les documents officiels seront indexés.</div>'
            f'<div class="kpi-source">Sources prévues : {escape_text(t["sources_prevues"])}</div>'
        )
    return f'<div class="kpi-card theme-{theme}">{head}{body}</div>'


def _public_card(title, text, bullets, page_key, link_label, pages):
    with st.container(key=f"public-{page_key}"):
        items = "".join(f"<li>{escape_text(b)}</li>" for b in bullets)
        st.markdown(f'<h3 class="card-title">{escape_text(title)}</h3><p class="card-text">{escape_text(text)}</p>'
                    f'<ul class="card-list">{items}</ul>', unsafe_allow_html=True)
        st.page_link(pages[page_key], label=link_label)


def _theme_card_html(theme, documents):
    t = config.THEMES[theme]
    titles = [d.get("titre") or d["source"] for d in documents.values()]
    if titles:
        text = " · ".join(escape_text(x) for x in titles)
    else:
        text = f"Documents à venir : {escape_text(t['sources_prevues'])}"
    return (f'<div class="theme-card theme-{theme}"><div class="card-title">{escape_text(t["label"])}</div>'
            f'<div class="card-text">{text}</div></div>')


def _go_to_assistant():
    question = st.session_state.get("home-question", "").strip()
    if question:
        st.session_state["pending_question"] = question
        st.session_state["theme_choice"] = "tous"
        st.session_state["go-assistant"] = True


def render():
    from ui.navigation import PAGES

    if st.session_state.pop("go-assistant", False):
        st.switch_page(PAGES["assistant"])

    prepare_corpus()
    collections = get_collections(get_chroma(), config.THEMES)
    documents = {t: indexed_documents(c) for t, c in collections.items()}
    n_docs = sum(len(d) for d in documents.values())

    hero, kpis = st.columns([1.15, 1], gap="large")
    with hero:
        st.markdown(
            '<span class="hero-tag">Pour la recherche, l\'enseignement et l\'action</span>'
            '<h1 class="hero-title">Les données et les connaissances, au même endroit.</h1>'
            '<p class="hero-text">Tabac, cancer du sein, cancer du col de l\'utérus et paludisme au Sénégal : '
            "interrogez un corpus de documents officiels (ANSD, OMS, CIRC, PNLP, ministère de la Santé), "
            "avec la source et la page pour chaque réponse.</p>",
            unsafe_allow_html=True,
        )
        with st.form("home-search", border=False):
            with st.container(key="home-search-bar", horizontal=True, vertical_alignment="center"):
                st.text_input("Question", key="home-question", label_visibility="collapsed",
                              placeholder="Ex. : évolution de la prévalence du tabac chez les 15-24 ans")
                st.form_submit_button("Interroger", type="primary", on_click=_go_to_assistant)
        st.markdown(f'<p class="hero-caption">{n_docs} document{"s" if n_docs > 1 else ""} '
                    f'indexé{"s" if n_docs > 1 else ""}</p>', unsafe_allow_html=True)

    with kpis:
        themes = list(config.THEMES)
        for pair in (themes[:2], themes[2:]):
            cols = st.columns(2)
            for col, theme in zip(cols, pair):
                col.markdown(_kpi_card_html(theme), unsafe_allow_html=True)
        with st.container(key="cta-data"):
            st.markdown('<div class="cta-title">Toutes les données, sources et méthodes</div>', unsafe_allow_html=True)
            st.page_link(PAGES["espace"], label="Espace données →")

    st.markdown('<h2 class="section-title">Pour qui ?</h2>', unsafe_allow_html=True)
    for col, public in zip(st.columns(3), PUBLICS):
        with col:
            _public_card(*public, PAGES)

    st.markdown('<h2 class="section-title">Quatre thèmes</h2>', unsafe_allow_html=True)
    for col, theme in zip(st.columns(4), config.THEMES):
        col.markdown(_theme_card_html(theme, documents[theme]), unsafe_allow_html=True)

    producers = sorted({d.get("producteur") for docs in documents.values() for d in docs.values() if d.get("producteur")})
    st.markdown(
        '<div class="site-footer">'
        f'<span>Sources du corpus : {escape_text(" · ".join(producers) or "aucune pour l’instant")}</span>'
        f'<span>Projet open source · {escape_text(config.DEPOT_GITHUB)}</span></div>',
        unsafe_allow_html=True,
    )
