"""
Registre des sources (documents/sources.json) : titre, producteur, année,
licence et type de chaque PDF. Ces informations sont enregistrées avec chaque
morceau dans ChromaDB pour filtrer par type de source et citer proprement.
"""

import json

import config

REQUIRED_FIELDS = ("fichier", "themes", "titre", "producteur", "annee", "licence", "type_source", "url", "publiable")


class RegistryError(ValueError):
    """Le fichier sources.json est mal formé."""


def _check_entry(entry, i, name):
    missing = [f for f in REQUIRED_FIELDS if entry.get(f) in (None, "", [])]
    if missing:
        raise RegistryError(f"{name}, document n°{i} : champ(s) manquant(s) : {', '.join(missing)}")
    fichier = entry["fichier"]
    themes = entry["themes"]
    if not isinstance(themes, list):
        raise RegistryError(f"{name}, {fichier} : « themes » doit être une liste, par exemple [\"cancer_col\"]")
    unknown = [t for t in themes if t not in config.THEMES]
    if unknown:
        raise RegistryError(f"{name}, {fichier} : thème(s) inconnu(s) {unknown} (valeurs possibles : {', '.join(config.THEMES)})")
    if len(set(themes)) != len(themes):
        raise RegistryError(f"{name}, {fichier} : thème en double dans {themes}")
    if entry["type_source"] not in config.SOURCE_TYPES:
        raise RegistryError(
            f"{name}, {fichier} : type_source {entry['type_source']!r} inconnu "
            f"(valeurs possibles : {', '.join(config.SOURCE_TYPES)})"
        )
    if not isinstance(entry["publiable"], bool):
        raise RegistryError(f"{name}, {fichier} : « publiable » doit valoir true ou false")


def load_registry(path=config.SOURCES_REGISTRY):
    """
    Renvoie {fichier: entrée}. Un registre absent donne un registre vide ;
    un registre mal formé lève RegistryError avec un message clair.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise RegistryError(f"{path.name} n'est pas un JSON valide (ligne {e.lineno}) : {e.msg}") from e

    registry = {}
    for i, entry in enumerate(data.get("documents", []), start=1):
        _check_entry(entry, i, path.name)
        if entry["fichier"] in registry:
            raise RegistryError(f"{path.name} : {entry['fichier']} est déclaré deux fois")
        registry[entry["fichier"]] = entry
    return registry


def home_theme(entry):
    """Dossier où le fichier est rangé : celui de son premier thème."""
    return entry["themes"][0]


def document_path(entry):
    return config.theme_docs_dir(home_theme(entry)) / entry["fichier"]


def entries_for_theme(registry, theme):
    """Documents du registre à indexer dans la collection de ce thème."""
    return [e for e in registry.values() if theme in e["themes"]]


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


def missing_file_warning(entry):
    return (
        f"⚠ {entry['fichier']} est décrit dans documents/sources.json mais introuvable dans "
        f"documents/{home_theme(entry)}/ (dossier de son premier thème) : il n'est pas indexé."
    )


def misplaced_warning(theme, entry):
    return (
        f"⚠ {entry['fichier']} est rangé dans documents/{theme}/, mais sources.json le place dans "
        f"documents/{home_theme(entry)}/ (premier de ses thèmes) : déplacez-le, il n'est pas indexé depuis ce dossier."
    )


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
