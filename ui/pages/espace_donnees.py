"""
Page Espace données : un onglet par thème, indicateurs chiffrés, sources et export CSV.

Les chiffres viennent de indicateurs.py (saisis à la main depuis les rapports,
avec leur page) : rien n'est produit par le modèle de langage. Un thème sans
indicateur affiche « données à venir ».
"""

import pandas as pd
import streamlit as st

import config
from indicateurs import INDICATEURS, format_number, get_value, rows_by_group
from rag.sources import RegistryError, load_registry
from ui.formatting import escape_text

CSV_COLUMNS = {
    "groupe": "Groupe", "indicateur": "Indicateur", "sous_groupe": "Population / sous-groupe",
    "annee": "Année", "valeur": "Valeur", "unite": "Unité", "source": "Source", "page": "Page",
}


# ---------------------------------------------------------------- blocs HTML (texte échappé)
def _kpi_html(label, value, note, note_class=""):
    return (f'<div class="data-kpi"><div class="data-kpi-label">{escape_text(label)}</div>'
            f'<div class="data-kpi-value">{value}</div>'
            f'<div class="data-kpi-note {note_class}">{escape_text(note)}</div></div>')


def _bars_html(rows, scale=None):
    """Barres horizontales : une par sous-groupe, longueur relative à `scale` (par défaut le maximum)."""
    scale = scale or max(r["valeur"] for r in rows)
    lines = "".join(
        f'<div class="bar-row"><span class="bar-label">{escape_text(r["sous_groupe"])}</span>'
        f'<span class="bar-track"><span class="bar-fill" style="width:{r["valeur"] / scale * 100:.1f}%"></span></span>'
        f'<span class="bar-value">{format_number(r["valeur"])} {escape_text(r["unite"])}</span></div>'
        for r in rows
    )
    return f'<div class="bars">{lines}</div>'


def _evolution_html(theme, indicators):
    blocks = []
    for name in indicators:
        before, after = get_value(theme, name, 2015), get_value(theme, name, 2023)
        blocks.append(
            f'<div class="evo-block"><div class="evo-head"><span>{escape_text(name)}</span>'
            f'<span><span class="evo-before">{format_number(before["valeur"])} %</span> → '
            f'<strong>{format_number(after["valeur"])} %</strong></span></div>'
            f'<span class="evo-bar evo-2015" style="width:{before["valeur"]:.1f}%"></span>'
            f'<span class="evo-bar evo-2023" style="width:{after["valeur"]:.1f}%"></span></div>'
        )
    legend = ('<div class="evo-legend"><span class="dot dot-2015"></span>2015 '
              '<span class="dot dot-2023"></span>2023</div>')
    return "".join(blocks) + legend


def _source_html(rows):
    pages = sorted({r["page"] for r in rows}, key=lambda p: (len(p), p))
    label = "page" if len(pages) == 1 else "pages"
    return f'<div class="data-source">Source : {escape_text(rows[0]["source"])}, {label} {escape_text(", ".join(pages))}</div>'


def _card(key, title, body_html, theme="tabac"):
    with st.container(key=f"data-card-{key}"):
        st.markdown(f'<div class="theme-{theme}"><div class="data-card-title">{escape_text(title)}</div>'
                    f"{body_html}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- contenu des onglets
def _render_tabac():
    t = "tabac"
    prev = get_value(t, "Prévalence globale", 2023)
    prev15 = get_value(t, "Prévalence globale", 2015)
    hommes, femmes = get_value(t, "Prévalence", 2023, "Hommes"), get_value(t, "Prévalence", 2023, "Femmes")
    ecig = get_value(t, "Connaissance de la cigarette électronique", 2023)
    ecig_usage = get_value(t, "Usage actuel de la cigarette électronique", 2023)
    arret, arret15 = get_value(t, "Intention d'arrêter", 2023), get_value(t, "Intention d'arrêter", 2015)
    prix = get_value(t, "Prix moyen d'un paquet", 2023)
    prix_evo = get_value(t, "Évolution du prix moyen d'un paquet depuis 2015", 2023)

    ecart = format_number(round(prev15["valeur"] - prev["valeur"], 1))
    kpis = [
        _kpi_html("Prévalence chez les adultes", f'{format_number(prev["valeur"])} %',
                  f"−{ecart} point depuis 2015", "note-good"),
        _kpi_html("Hommes / femmes",
                  f'{format_number(hommes["valeur"])} % <span class="kpi-secondary">/ {format_number(femmes["valeur"])} %</span>',
                  f"GATS 2023, page {hommes['page']}"),
        _kpi_html("Connaissent la cigarette électronique", f'{format_number(ecig["valeur"])} %',
                  f"Usage actuel : {format_number(ecig_usage['valeur'])} % (pages {ecig_usage['page']})"),
        _kpi_html("Fumeurs voulant arrêter", f'{format_number(arret["valeur"])} %',
                  f"{format_number(arret15['valeur'])} % en 2015", "note-bad"),
    ]
    for col, kpi in zip(st.columns(4), kpis):
        col.markdown(kpi, unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    with left:
        age = rows_by_group(t, "Prévalence par âge")
        _card("age", "Prévalence par tranche d'âge", _bars_html(age) + _source_html(age))
    with right:
        tend = rows_by_group(t, "Tendances")
        _card("evolution", "Évolution 2015 → 2023",
              _evolution_html(t, ["Prévalence globale", "Intention d'arrêter", "Tentative d'arrêt"]) + _source_html(tend))

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        milieu = rows_by_group(t, "Prévalence par milieu")
        _card("milieu", "Prévalence par milieu de résidence", _bars_html(milieu) + _source_html(milieu))
    with c2:
        statut = rows_by_group(t, "Statut tabagique")
        _card("statut", "Statut tabagique des adultes", _bars_html(statut, scale=100) + _source_html(statut))
    with c3:
        _card("prix", "Prix moyen d'un paquet",
              f'<div class="data-kpi-value">{format_number(prix["valeur"])} {escape_text(prix["unite"])}</div>'
              f'<div class="data-kpi-note">+{format_number(prix_evo["valeur"])} % depuis 2015</div>'
              + _source_html([prix, prix_evo]))


def _render_pending(theme):
    t = config.THEMES[theme]
    st.markdown(
        f'<div class="data-pending theme-{theme}"><div class="data-pending-title">Données à venir</div>'
        "<p>Les indicateurs de ce thème seront publiés quand les documents officiels auront été rassemblés, "
        "vérifiés et indexés. Aucun chiffre n'est affiché en attendant.</p>"
        f'<p class="data-source">Sources prévues : {escape_text(t["sources_prevues"])}</p></div>',
        unsafe_allow_html=True,
    )


def _sources_table_html(theme, registry):
    entries = [e for (th, _), e in registry.items() if th == theme]
    if not entries:
        return '<p class="data-source">Aucun document répertorié pour ce thème dans documents/sources.json.</p>'
    rows = "".join(
        f"<tr><td><strong>{escape_text(e.get('titre_court') or e['titre'])}</strong></td>"
        f"<td>{escape_text(e['producteur'])}</td><td>{escape_text(e['annee'])}</td>"
        f"<td>{escape_text(e['licence'])}</td></tr>"
        for e in sorted(entries, key=lambda e: e["annee"])
    )
    return ('<table class="sources-table"><thead><tr><th>Document</th><th>Producteur</th><th>Année</th>'
            f"<th>Licence</th></tr></thead><tbody>{rows}</tbody></table>")


def indicators_csv(theme):
    """CSV des indicateurs affichés (séparateur ; et virgule décimale, lisible par Excel en français)."""
    frame = pd.DataFrame(INDICATEURS[theme], columns=list(CSV_COLUMNS)).rename(columns=CSV_COLUMNS)
    return frame.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def _sources_section(theme, registry):
    with st.container(key=f"data-card-sources-{theme}"):
        head, button = st.columns([3, 1.3], vertical_alignment="center")
        head.markdown('<div class="data-card-title">Sources et téléchargements</div>', unsafe_allow_html=True)
        if INDICATEURS[theme]:
            button.download_button(
                "Télécharger les données (CSV)", data=indicators_csv(theme), icon=":material/download:",
                file_name=f"indicateurs_{theme}.csv", mime="text/csv", type="primary", key=f"csv-{theme}",
            )
        st.markdown(_sources_table_html(theme, registry), unsafe_allow_html=True)


# ---------------------------------------------------------------- page
TITRES = {
    "tabac": ("Tabac au Sénégal",
              "Indicateurs de l'enquête GATS 2023 · chiffres saisis à la main depuis le rapport, page par page"),
    "cancer_sein": ("Cancer du sein au Sénégal", "Indicateurs à venir"),
    "cancer_col": ("Cancer du col de l'utérus au Sénégal", "Indicateurs à venir"),
    "paludisme": ("Paludisme au Sénégal", "Indicateurs à venir"),
}


def render():
    try:
        registry = load_registry()
    except RegistryError as e:
        st.error(f"Registre des sources illisible : {e}")
        registry = {}

    with st.container(key="data-tabs"):
        tabs = st.tabs([config.THEMES[t]["label_court"] for t in config.THEMES])
    for tab, theme in zip(tabs, config.THEMES):
        with tab:
            title, subtitle = TITRES[theme]
            st.markdown(f'<h1 class="data-title">{escape_text(title)}</h1>'
                        f'<p class="data-subtitle">{escape_text(subtitle)}</p>', unsafe_allow_html=True)
            if theme == "tabac":
                _render_tabac()
            else:
                _render_pending(theme)
            _sources_section(theme, registry)
