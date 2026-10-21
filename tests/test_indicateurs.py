"""
Les chiffres du tableau de bord GATS 2023 ont été vérifiés à la main dans le
rapport. Ce test les fige : il échoue si une valeur ou une page source change.
"""

import config
from indicateurs import INDICATEURS, format_number, get_value

# Copie exacte du tableau de bord de l'étape 1 : (indicateur, sous-groupe, année) -> (valeur, unité, page)
VERIFIED_GATS_2023 = {
    ("Prévalence globale", "Adultes", 2015): (6.0, "%", "94"),
    ("Prévalence globale", "Adultes", 2023): (4.4, "%", "94"),
    ("Intention d'arrêter", "Fumeurs", 2015): (79.8, "%", "94"),
    ("Intention d'arrêter", "Fumeurs", 2023): (62.5, "%", "94"),
    ("Tentative d'arrêt", "Fumeurs", 2015): (59.6, "%", "94"),
    ("Tentative d'arrêt", "Fumeurs", 2023): (52.4, "%", "94"),
    ("Connaissance de la cigarette électronique", "Adultes", 2023): (36.6, "%", "94"),
    ("Usage actuel de la cigarette électronique", "Adultes", 2023): (0.4, "%", "15, 94"),
    ("Prix moyen d'un paquet", "Ensemble", 2023): (849, "FCFA", "94"),
    ("Évolution du prix moyen d'un paquet depuis 2015", "Ensemble", 2023): (21.2, "%", "94"),
    ("Prévalence", "15-24 ans", 2023): (1.6, "%", "49-50"),
    ("Prévalence", "25-44 ans", 2023): (6.2, "%", "49-50"),
    ("Prévalence", "45-64 ans", 2023): (6.3, "%", "49-50"),
    ("Prévalence", "65 ans et +", 2023): (3.0, "%", "49-50"),
    ("Prévalence", "Hommes", 2023): (8.2, "%", "50 (tableau 4.10)"),
    ("Prévalence", "Femmes", 2023): (0.8, "%", "50 (tableau 4.10)"),
    ("Prévalence", "Urbain", 2023): (5.0, "%", "49"),
    ("Prévalence", "Rural", 2023): (3.9, "%", "49"),
    ("Part des adultes", "Non-fumeurs", 2023): (95.9, "%", "37 (graphique 4.1)"),
    ("Part des adultes", "Fumeurs quotidiens", 2023): (3.4, "%", "37 (graphique 4.1)"),
    ("Part des adultes", "Fumeurs occasionnels", 2023): (0.7, "%", "37 (graphique 4.1)"),
}


def test_gats_figures_are_unchanged():
    actual = {
        (r["indicateur"], r["sous_groupe"], r["annee"]): (r["valeur"], r["unite"], r["page"])
        for r in INDICATEURS["tabac"]
    }
    assert actual == VERIFIED_GATS_2023


def test_every_indicator_has_a_source_and_page():
    for theme, rows in INDICATEURS.items():
        for row in rows:
            assert row["source"] and row["page"], row


def test_no_figure_for_themes_without_documents():
    for theme in ("cancer_sein", "cancer_col", "paludisme"):
        assert INDICATEURS[theme] == []
    assert set(INDICATEURS) == set(config.THEMES)


def test_get_value_and_format():
    assert format_number(get_value("tabac", "Prévalence globale", 2015)["valeur"]) == "6,0"
    assert format_number(get_value("tabac", "Prévalence globale", 2023)["valeur"]) == "4,4"
    assert format_number(849) == "849"
    assert get_value("paludisme", "Prévalence globale", 2023) is None


def test_csv_export_contains_displayed_indicators_with_sources():
    from ui.pages.espace_donnees import indicators_csv

    text = indicators_csv("tabac").decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0] == "Groupe;Indicateur;Population / sous-groupe;Année;Valeur;Unité;Source;Page"
    assert len(lines) == 1 + len(INDICATEURS["tabac"])
    assert "Tendances;Prévalence globale;Adultes;2023;4,4;%;GATS Sénégal 2023 (ANSD);94" in lines
    assert "Prévalence par sexe;Prévalence;Hommes;2023;8,2;%;GATS Sénégal 2023 (ANSD);50 (tableau 4.10)" in lines
    assert indicators_csv("paludisme").decode("utf-8-sig").strip() == lines[0]
