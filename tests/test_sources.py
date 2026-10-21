import json

import pytest

import config
from rag.pipeline import answer_question
from rag.retrieval import search, source_type_filter
from rag.sources import RegistryError, chunk_metadata, full_reference, load_registry
from rag.store import index_theme, sync_source_metadata
from tests.conftest import FakeLLM

ENTRY = {
    "fichier": "rapport.pdf", "theme": "paludisme", "titre": "Rapport complet", "titre_court": "Rapport",
    "producteur": "PNLP", "annee": 2024, "licence": "Licence X", "type_source": "pnlp",
}


def write_registry(tmp_path, documents):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"documents": documents}, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- registre
def test_real_registry_describes_the_two_tabac_pdfs():
    registry = load_registry()
    for fichier in ("GATS_2023.pdf", "Rapport OMS_Tabagisme 2000_2024  projections 2025_2030.pdf"):
        assert ("tabac", fichier) in registry
        assert (config.theme_docs_dir("tabac") / fichier).exists()
    assert registry[("tabac", "GATS_2023.pdf")]["type_source"] == "enquete_nationale"
    assert registry[("tabac", "Rapport OMS_Tabagisme 2000_2024  projections 2025_2030.pdf")]["licence"] == "CC BY-NC-SA 3.0 IGO"


def test_every_pdf_on_disk_is_in_the_registry():
    registry = load_registry()
    for theme in config.THEMES:
        for pdf in config.theme_docs_dir(theme).glob("*.pdf"):
            assert (theme, pdf.name) in registry, f"{theme}/{pdf.name} absent de sources.json"


def test_load_registry(tmp_path):
    registry = load_registry(write_registry(tmp_path, [ENTRY]))
    assert registry[("paludisme", "rapport.pdf")]["producteur"] == "PNLP"


def test_missing_registry_file_gives_empty_registry(tmp_path):
    assert load_registry(tmp_path / "absent.json") == {}


@pytest.mark.parametrize("change, message", [
    ({"titre": ""}, "champ"),
    ({"type_source": "blog"}, "type_source"),
    ({"theme": "grippe"}, "thème inconnu"),
])
def test_invalid_registry_raises_clear_error(tmp_path, change, message):
    with pytest.raises(RegistryError, match=message):
        load_registry(write_registry(tmp_path, [ENTRY | change]))


def test_malformed_json_raises_clear_error(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text("{ pas du json", encoding="utf-8")
    with pytest.raises(RegistryError, match="JSON"):
        load_registry(path)


def test_chunk_metadata_and_full_reference():
    meta = chunk_metadata(ENTRY, "rapport.pdf")
    assert meta == {"titre": "Rapport", "titre_complet": "Rapport complet", "producteur": "PNLP",
                    "annee": 2024, "licence": "Licence X", "type_source": "pnlp"}
    assert full_reference(meta, 7) == "PNLP. Rapport complet. 2024, p. 7."


def test_unregistered_document_gets_neutral_metadata():
    meta = chunk_metadata(None, "inconnu.pdf")
    assert meta["type_source"] == config.UNREGISTERED_SOURCE_TYPE
    assert None not in meta.values()  # ChromaDB refuse les valeurs None
    assert full_reference(meta, 3) == "inconnu.pdf, p. 3."


# ---------------------------------------------------------------- indexation
class RecordingCollection:
    """Collection minimale pour tester l'indexation sans ChromaDB."""

    def __init__(self, metadatas=()):
        self.ids = [f"id{i}" for i in range(len(metadatas))]
        self.metadatas = list(metadatas)

    def count(self):
        return len(self.ids)

    def get(self, include=None):
        return {"ids": list(self.ids), "metadatas": [dict(m) for m in self.metadatas]}

    def update(self, ids, metadatas):
        self.metadatas = [dict(m) for m in metadatas]

    def add(self, documents, embeddings, metadatas, ids):
        self.ids += ids
        self.metadatas += metadatas


class ArrayLike(list):
    def tolist(self):
        return list(self)


class Embedder:
    def encode(self, texts, **kwargs):
        return ArrayLike([[0.0, 1.0] for _ in texts])


def test_pdf_missing_from_registry_warns_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path)
    (tmp_path / "paludisme").mkdir()
    (tmp_path / "paludisme" / "inconnu.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr("rag.store.extract_pages", lambda path: [(1, "Une page de texte.")])

    messages, collection = [], RecordingCollection()
    added = index_theme("paludisme", Embedder(), collection, log=messages.append, registry={})

    assert added == 1
    assert any("inconnu.pdf" in m and "sources.json" in m for m in messages)
    assert collection.metadatas[0]["type_source"] == config.UNREGISTERED_SOURCE_TYPE


def test_registered_pdf_metadata_is_stored_at_indexing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path)
    (tmp_path / "paludisme").mkdir()
    (tmp_path / "paludisme" / "rapport.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr("rag.store.extract_pages", lambda path: [(4, "Texte de la page quatre.")])

    collection, messages = RecordingCollection(), []
    index_theme("paludisme", Embedder(), collection, log=messages.append,
                registry={("paludisme", "rapport.pdf"): ENTRY})

    meta = collection.metadatas[0]
    assert meta["type_source"] == "pnlp" and meta["titre"] == "Rapport" and meta["page"] == 4
    assert not any(m.startswith("⚠") for m in messages)


def test_sync_updates_already_indexed_chunks():
    collection = RecordingCollection([{"source": "rapport.pdf", "page": 1, "theme": "paludisme"}])
    sync_source_metadata("paludisme", collection, {("paludisme", "rapport.pdf"): ENTRY})
    assert collection.metadatas[0]["type_source"] == "pnlp"
    assert collection.metadatas[0]["page"] == 1


# ---------------------------------------------------------------- filtre par type de source
def test_source_type_filter():
    assert source_type_filter(None) is None
    assert source_type_filter(["oms", "enquete_nationale"]) == {"type_source": {"$in": ["enquete_nationale", "oms"]}}


def test_search_keeps_only_checked_source_types(embedder, tabac_collection):
    hits = search("q", embedder, {"tabac": tabac_collection}, source_types=["oms"])
    assert hits and all(h["type_source"] == "oms" for h in hits)

    hits = search("q", embedder, {"tabac": tabac_collection}, source_types=["enquete_nationale"])
    assert {h["source"] for h in hits} == {"GATS_2023.pdf"}


def test_search_without_filter_returns_all_types(embedder, tabac_collection):
    hits = search("q", embedder, {"tabac": tabac_collection}, top_k=4)
    assert {h["type_source"] for h in hits} == {"enquete_nationale", "oms"}


def test_no_checked_type_means_no_result_and_no_llm_call(embedder, tabac_collection):
    llm = FakeLLM()
    result = answer_question("q", embedder, {"tabac": tabac_collection}, llm, source_types=[])
    assert result.hits == [] and result.guardrail_triggered
    assert llm.calls == []


def test_pipeline_passes_filter_to_search(embedder, tabac_collection):
    result = answer_question("q", embedder, {"tabac": tabac_collection}, FakeLLM(), source_types=["oms"])
    assert [c["titre"] for c in result.citations] == ["OMS tabac"]
