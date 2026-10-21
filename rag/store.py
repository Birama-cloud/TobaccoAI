"""
Embeddings locaux (sentence-transformers) et base vectorielle ChromaDB.
Une collection par thème : tabac, cancer_sein, cancer_col.
"""

import config
from rag.chunking import build_chunks
from rag.pdf import extract_pages
from rag.sources import (
    chunk_metadata, document_path, entries_for_theme, load_registry, misplaced_warning,
    missing_file_warning, unregistered_warning,
)


def load_embedder(model_name=config.EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_chroma_client(persist_dir=config.CHROMA_PERSIST_DIR):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    persist_dir.mkdir(parents=True, exist_ok=True)
    # Télémétrie désactivée : rien n'est envoyé à ChromaDB sur l'utilisation.
    return chromadb.PersistentClient(
        path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )


def get_collection(client, theme):
    # distance cosinus : plus interprétable pour un seuil de pertinence que L2 brut
    return client.get_or_create_collection(
        name=config.THEMES[theme]["collection"],
        metadata={"hnsw:space": "cosine"},
    )


def get_collections(client, themes):
    """Renvoie {thème: collection} pour les thèmes demandés."""
    return {theme: get_collection(client, theme) for theme in themes}


def list_pdf_files(theme):
    """Liste les PDF présents dans le dossier du thème."""
    folder = config.theme_docs_dir(theme)
    if not folder.is_dir():
        return []
    return sorted(f.name for f in folder.iterdir() if f.suffix.lower() == ".pdf")


def get_indexed_sources(collection):
    """Renvoie {nom_de_fichier: nombre_de_morceaux} pour tout ce qui est déjà indexé."""
    if collection.count() == 0:
        return {}
    counts = {}
    for meta in collection.get(include=["metadatas"])["metadatas"]:
        src = meta.get("source", "inconnu")
        counts[src] = counts.get(src, 0) + 1
    return counts


def indexed_documents(collection):
    """Renvoie {nom_de_fichier: métadonnées d'un de ses morceaux} pour les documents indexés."""
    if collection.count() == 0:
        return {}
    docs = {}
    for meta in collection.get(include=["metadatas"])["metadatas"]:
        docs.setdefault(meta.get("source", "inconnu"), meta)
    return docs


def sync_source_metadata(theme, collection, registry):
    """
    Recopie les informations du registre dans les morceaux déjà indexés (titre,
    producteur, type de source...). Pas besoin de recalculer les embeddings
    quand on corrige ou complète sources.json.
    """
    if collection.count() == 0:
        return
    data = collection.get(include=["metadatas"])
    new_metadatas = []
    for meta in data["metadatas"]:
        entry = registry.get(meta.get("source"))
        new_metadatas.append({**meta, **chunk_metadata(entry, meta.get("source"))})
    if new_metadatas != data["metadatas"]:
        collection.update(ids=data["ids"], metadatas=new_metadatas)


def documents_for_theme(theme, registry, log=print):
    """
    Documents à indexer dans la collection d'un thème, sous forme {fichier: chemin} :
    - ceux du registre qui listent ce thème (rangés dans le dossier de leur premier thème) ;
    - les PDF du dossier du thème absents du registre (indexés avec un avertissement).
    """
    documents = {}
    for entry in entries_for_theme(registry, theme):
        path = document_path(entry)
        if path.exists():
            documents[entry["fichier"]] = path
        else:
            log(missing_file_warning(entry))
    for fname in list_pdf_files(theme):
        entry = registry.get(fname)
        if entry is None:
            log(unregistered_warning(theme, fname))
            documents[fname] = config.theme_docs_dir(theme) / fname
        elif document_path(entry) != config.theme_docs_dir(theme) / fname:
            log(misplaced_warning(theme, entry))
    return documents


def index_theme(theme, embedder, collection, force_reindex=False, log=print, registry=None):
    """
    Extraction -> découpage par page -> embeddings -> stockage, pour tous les
    documents du thème (voir documents_for_theme). Un document à plusieurs
    thèmes est indexé dans la collection de chacun. Ne traite que les documents
    pas encore indexés, sauf si force_reindex=True (repart alors de zéro pour ce thème).
    Renvoie le nombre de morceaux ajoutés. `log` reçoit les messages de suivi.
    """
    if registry is None:
        registry = load_registry()

    if force_reindex and collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])

    documents = documents_for_theme(theme, registry, log=log)
    sync_source_metadata(theme, collection, registry)

    already_indexed = get_indexed_sources(collection)
    files_to_process = [f for f in sorted(documents) if f not in already_indexed]
    label = config.THEMES[theme]["label"]

    total_added = 0
    for fname in files_to_process:
        try:
            pages = extract_pages(documents[fname])
        except Exception as e:
            log(f"[{label}] Erreur d'extraction de {fname} : {e}")
            continue
        if not pages:
            log(f"[{label}] Aucun texte extrait de {fname} — ignoré.")
            continue

        extra = chunk_metadata(registry.get(fname), fname)
        texts, metadatas, ids = build_chunks(pages, source=fname, theme=theme, extra_metadata=extra)
        if not texts:
            continue

        embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        collection.add(documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)
        total_added += len(texts)
        log(f"[{label}] {fname} indexé : {len(texts)} morceaux sur {len(pages)} pages.")

    return total_added
