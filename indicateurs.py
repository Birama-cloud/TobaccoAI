"""
Indicateurs chiffrés affichés sur la plateforme (Accueil, Espace données, export CSV).

Ces chiffres ne sont PAS produits par le modèle de langage : ils ont été saisis
à la main depuis les rapports, avec la page source de chacun. Ceux du tabac
reprennent à l'identique le tableau de bord GATS 2023 de l'étape 1.
Ne jamais modifier une valeur sans l'avoir revérifiée dans le document.

Un thème sans entrée affiche « données à venir » : aucun chiffre n'est inventé.
"""

GATS_2023 = "GATS Sénégal 2023 (ANSD)"


def _row(groupe, indicateur, sous_groupe, annee, valeur, unite, page):
    return {
        "groupe": groupe,
        "indicateur": indicateur,
        "sous_groupe": sous_groupe,
        "annee": annee,
        "valeur": valeur,
        "unite": unite,
        "source": GATS_2023,
        "page": page,
    }


INDICATEURS = {
    "tabac": [
        # Indicateurs clés et tendances 2015 -> 2023 : synthèse page 94
        _row("Tendances", "Prévalence globale", "Adultes", 2015, 6.0, "%", "94"),
        _row("Tendances", "Prévalence globale", "Adultes", 2023, 4.4, "%", "94"),
        _row("Tendances", "Intention d'arrêter", "Fumeurs", 2015, 79.8, "%", "94"),
        _row("Tendances", "Intention d'arrêter", "Fumeurs", 2023, 62.5, "%", "94"),
        _row("Tendances", "Tentative d'arrêt", "Fumeurs", 2015, 59.6, "%", "94"),
        _row("Tendances", "Tentative d'arrêt", "Fumeurs", 2023, 52.4, "%", "94"),
        _row("Cigarette électronique", "Connaissance de la cigarette électronique", "Adultes", 2023, 36.6, "%", "94"),
        _row("Cigarette électronique", "Usage actuel de la cigarette électronique", "Adultes", 2023, 0.4, "%", "15, 94"),
        _row("Prix", "Prix moyen d'un paquet", "Ensemble", 2023, 849, "FCFA", "94"),
        _row("Prix", "Évolution du prix moyen d'un paquet depuis 2015", "Ensemble", 2023, 21.2, "%", "94"),
        # Prévalence par sous-groupe
        _row("Prévalence par âge", "Prévalence", "15-24 ans", 2023, 1.6, "%", "49-50"),
        _row("Prévalence par âge", "Prévalence", "25-44 ans", 2023, 6.2, "%", "49-50"),
        _row("Prévalence par âge", "Prévalence", "45-64 ans", 2023, 6.3, "%", "49-50"),
        _row("Prévalence par âge", "Prévalence", "65 ans et +", 2023, 3.0, "%", "49-50"),
        _row("Prévalence par sexe", "Prévalence", "Hommes", 2023, 8.2, "%", "50 (tableau 4.10)"),
        _row("Prévalence par sexe", "Prévalence", "Femmes", 2023, 0.8, "%", "50 (tableau 4.10)"),
        _row("Prévalence par milieu", "Prévalence", "Urbain", 2023, 5.0, "%", "49"),
        _row("Prévalence par milieu", "Prévalence", "Rural", 2023, 3.9, "%", "49"),
        _row("Statut tabagique", "Part des adultes", "Non-fumeurs", 2023, 95.9, "%", "37 (graphique 4.1)"),
        _row("Statut tabagique", "Part des adultes", "Fumeurs quotidiens", 2023, 3.4, "%", "37 (graphique 4.1)"),
        _row("Statut tabagique", "Part des adultes", "Fumeurs occasionnels", 2023, 0.7, "%", "37 (graphique 4.1)"),
    ],
    "cancer_sein": [],
    "cancer_col": [],
    "paludisme": [],
}


def get_value(theme, indicateur, annee, sous_groupe=None):
    """Renvoie la ligne d'un indicateur, ou None s'il n'existe pas."""
    for row in INDICATEURS.get(theme, []):
        if row["indicateur"] == indicateur and row["annee"] == annee and (
            sous_groupe is None or row["sous_groupe"] == sous_groupe
        ):
            return row
    return None


def rows_by_group(theme, groupe):
    return [row for row in INDICATEURS.get(theme, []) if row["groupe"] == groupe]


def format_number(value):
    """4.4 -> '4,4' ; 6.0 -> '6,0' ; 849 -> '849' (format français, une décimale comme le rapport)."""
    text = f"{value:.1f}" if isinstance(value, float) else str(value)
    return text.replace(".", ",")
