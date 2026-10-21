import config
from rag.retrieval import filter_relevant, normalize_query, search
from tests.conftest import FakeCollection


def test_normalize_query():
    assert normalize_query("  QUEL est   le TAUX ?  ") == "quel est le taux ?"


def test_threshold_keeps_only_close_hits():
    hits = [{"distance": 0.2}, {"distance": config.RELEVANCE_DISTANCE_THRESHOLD}, {"distance": 0.79}]
    kept = filter_relevant(hits)
    assert [h["distance"] for h in kept] == [0.2, config.RELEVANCE_DISTANCE_THRESHOLD]


def test_threshold_rejects_everything_when_off_topic():
    assert filter_relevant([{"distance": 1.1}, {"distance": 1.3}]) == []


def test_search_returns_hits_sorted_with_metadata(embedder, tabac_collection):
    hits = search("taux", embedder, {"tabac": tabac_collection}, top_k=2)
    assert [(h["source"], h["page"]) for h in hits] == [("GATS_2023.pdf", 14), ("GATS_2023.pdf", 49)]
    assert hits[0]["theme"] == "tabac"


def test_search_merges_themes_and_skips_empty_collections(embedder, tabac_collection):
    sein = FakeCollection([("Dépistage.", {"source": "sein.pdf", "page": 2, "theme": "cancer_sein"}, 0.40)])
    collections = {"tabac": tabac_collection, "cancer_sein": sein, "cancer_col": FakeCollection([])}

    hits = search("question", embedder, collections, top_k=3)

    assert [h["distance"] for h in hits] == [0.30, 0.40, 0.50]
    assert hits[1]["theme"] == "cancer_sein"


def test_search_with_no_documents(embedder):
    assert search("question", embedder, {"cancer_col": FakeCollection([])}) == []
