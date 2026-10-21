"""Recherche des morceaux les plus pertinents, sur un ou plusieurs thèmes."""

import config


def normalize_query(query):
    """
    Normalisation légère avant l'embedding : une question TOUT EN MAJUSCULES
    produit un vecteur plus éloigné du texte indexé. On ne touche pas au texte
    affiché, seulement à la version envoyée au modèle d'embedding.
    """
    return " ".join(query.strip().lower().split())


def source_type_filter(source_types):
    """
    Filtre ChromaDB sur le type de source. None = pas de filtre (tous les
    documents, y compris ceux absents du registre).
    """
    if source_types is None:
        return None
    return {"type_source": {"$in": sorted(source_types)}}


def search(query, embedder, collections, top_k=config.TOP_K, source_types=None):
    """
    Cherche dans chaque collection (une par thème) puis garde les `top_k`
    morceaux les plus proches, tous thèmes confondus.
    `source_types` : types de source autorisés (None = tous).
    Renvoie une liste de dicts : {text, source, page, theme, distance, titre, ...}.
    """
    if source_types is not None and not source_types:
        return []  # aucune case cochée : aucun document autorisé

    query_embedding = embedder.encode([normalize_query(query)], normalize_embeddings=True)[0]
    query_embedding = list(map(float, query_embedding))
    where = source_type_filter(source_types)

    hits = []
    for theme, collection in collections.items():
        count = collection.count()
        if count == 0:
            continue
        results = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, count), where=where)
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            hits.append({
                **meta,
                "text": doc,
                "source": meta.get("source", "document"),
                "page": meta.get("page", "?"),
                "theme": meta.get("theme", theme),
                "titre": meta.get("titre") or meta.get("source", "document"),
                "distance": dist,
            })

    hits.sort(key=lambda h: h["distance"])
    return hits[:top_k]


def filter_relevant(hits, threshold=config.RELEVANCE_DISTANCE_THRESHOLD):
    """Garde-fou anti-hallucination : on écarte les morceaux trop éloignés de la question."""
    return [h for h in hits if h["distance"] <= threshold]
