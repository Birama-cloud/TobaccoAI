"""
Registre des sources (documents/sources.json) : titre, producteur, année,
licence et type de chaque PDF. Ces informations sont enregistrées avec chaque
morceau dans ChromaDB pour filtrer par type de source et citer proprement.
"""

import json

import config

REQUIRED_FIELDS = ("fichier", "theme", "titre", "producteur", "annee", "licence", "type_source")


class RegistryError(ValueError):
    """Le fichier sources.json est mal formé."""


def load_registry(path=config.SOURCES_REGISTRY):
    """
    Renvoie {(thème, fichier): entrée}. Un registre absent donne un registre
    vide ; un registre mal formé lève RegistryError avec un message clair.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise RegistryError(f"{path.name} n'est pas un JSON valide (ligne {e.lineno}) : {e.msg}") from e

    registry = {}
    for i, entry in enumerate(data.get("documents", []), start=1):
        missing = [f for f in REQUIRED_FIELDS if entry.get(f) in (None, "")]
        if missing:
            raise RegistryError(f"{path.name}, document n°{i} : champ(s) manquant(s) : {', '.join(missing)}")
        if entry["theme"] not in config.THEMES:
            raise RegistryError(f"{path.name}, {entry['fichier']} : thème inconnu {entry['theme']!r}")
        if entry["type_source"] not in config.SOURCE_TYPES:
            raise RegistryError(
                f"{path.name}, {entry['fichier']} : type_source {entry['type_source']!r} inconnu "
                f"(valeurs possibles : {', '.join(config.SOURCE_TYPES)})"
            )
        registry[(entry["theme"], entry["fichier"])] = entry
    return registry


def chunk_metadata(entry, fichier):
    """Métadonnées ChromaDB d'un document (valeurs simples uniquement, jamais None)."""
    if entry is None:
        return {
            "titre": fichier,
            "titre_complet": fichier,
            "producteur": "",
            "annee": 0,
            "licence": "",
            "type_source": config.UNREGISTERED_SOURCE_TYPE,
        }
    return {
        "titre": entry.get("titre_court") or entry["titre"],
        "titre_complet": entry["titre"],
        "producteur": entry["producteur"],
        "annee": int(entry["annee"]),
        "licence": entry["licence"],
        "type_source": entry["type_source"],
    }


def unregistered_warning(theme, fichier):
    return (
        f"⚠ {fichier} (thème {config.THEMES[theme]['label']}) n'est pas décrit dans documents/sources.json : "
        "il est indexé sans titre ni type de source, et n'apparaît donc pas quand on filtre par type de source. "
        "Ajoutez-le au registre puis relancez l'indexation."
    )


def full_reference(meta, page):
    """Référence complète à copier : producteur, titre, année, page."""
    parts = [meta.get("producteur"), meta.get("titre_complet") or meta.get("titre"), meta.get("annee") or None]
    text = ". ".join(str(p) for p in parts if p)
    return f"{text}, p. {page}."
