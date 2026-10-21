from rag.chunking import build_chunks, chunk_page_text, sanitize_id_prefix, split_into_sentences


def test_split_into_sentences():
    assert split_into_sentences("Première phrase. Deuxième ! Troisième ?") == [
        "Première phrase.", "Deuxième !", "Troisième ?"
    ]


def test_short_text_gives_one_chunk():
    assert chunk_page_text("Une phrase courte.", chunk_size=50, overlap=5) == ["Une phrase courte."]


def test_empty_text_gives_no_chunk():
    assert chunk_page_text("   ", chunk_size=50, overlap=5) == []


def test_chunks_respect_size_and_overlap():
    sentences = [" ".join(f"m{i}_{j}" for j in range(10)) + "." for i in range(6)]  # 6 phrases de 10 mots
    chunks = chunk_page_text(" ".join(sentences), chunk_size=25, overlap=5)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        # le morceau suivant reprend les 5 derniers mots du précédent
        assert current.split()[:5] == previous.split()[-5:]
    # aucun mot perdu
    all_words = set(" ".join(sentences).split())
    assert all_words == set(" ".join(chunks).split())


def test_chunks_never_mix_pages():
    pages = [(3, "Texte de la page trois."), (4, "Texte de la page quatre.")]
    texts, metadatas, ids = build_chunks(pages, source="rapport.pdf", theme="tabac")

    assert texts == ["Texte de la page trois.", "Texte de la page quatre."]
    assert metadatas == [
        {"source": "rapport.pdf", "page": 3, "theme": "tabac"},
        {"source": "rapport.pdf", "page": 4, "theme": "tabac"},
    ]
    assert len(set(ids)) == len(ids)


def test_sanitize_id_prefix():
    assert sanitize_id_prefix("Rapport OMS 2025.pdf") == "Rapport_OMS_2025_pdf"
