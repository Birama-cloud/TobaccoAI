"""Découpage du texte en morceaux — jamais à cheval sur deux pages."""

import re

import config


def split_into_sentences(text):
    """Découpage simple en phrases pour éviter de couper au milieu d'une idée."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_page_text(text, chunk_size=config.CHUNK_SIZE_WORDS, overlap=config.CHUNK_OVERLAP_WORDS):
    """
    Découpe le texte d'UNE page en morceaux avec chevauchement, en respectant
    autant que possible les frontières de phrases.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_words = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_words) + len(sentence_words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # chevauchement : on garde les derniers mots du morceau précédent
            current_words = current_words[-overlap:] if overlap else []
        current_words.extend(sentence_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def sanitize_id_prefix(filename):
    """Rend un nom de fichier utilisable comme préfixe d'identifiant ChromaDB."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', filename)


def build_chunks(pages, source, theme, extra_metadata=None):
    """
    Transforme les pages d'un document en (textes, métadonnées, identifiants)
    prêts à être indexés. Chaque morceau garde son document, sa page, son thème
    et les informations du registre des sources (extra_metadata).
    """
    id_prefix = sanitize_id_prefix(source)
    texts, metadatas, ids = [], [], []
    for page_number, page_text in pages:
        for chunk in chunk_page_text(page_text):
            texts.append(chunk)
            metadatas.append({**(extra_metadata or {}), "source": source, "page": page_number, "theme": theme})
            ids.append(f"{id_prefix}_{len(ids)}")
    return texts, metadatas, ids
