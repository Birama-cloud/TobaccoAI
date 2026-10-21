"""Faux objets pour tester le pipeline sans clé d'API, sans modèle ni base réelle."""

import pytest


GATS_META = {"source": "GATS_2023.pdf", "theme": "tabac", "titre": "GATS Sénégal 2023",
             "titre_complet": "Enquête GATS Sénégal 2023", "producteur": "ANSD", "annee": 2023,
             "type_source": "enquete_nationale"}
OMS_META = {"source": "OMS.pdf", "theme": "tabac", "titre": "OMS tabac", "titre_complet": "WHO report",
            "producteur": "OMS", "annee": 2025, "type_source": "oms"}


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeCollection:
    """Imite une collection ChromaDB : renvoie des résultats fixés à l'avance."""

    def __init__(self, entries):
        # entries : liste de (texte, métadonnées, distance)
        self.entries = sorted(entries, key=lambda e: e[2])

    def count(self):
        return len(self.entries)

    def query(self, query_embeddings, n_results, where=None):
        entries = self.entries
        if where is not None:  # seul le filtre utilisé par l'application : {"type_source": {"$in": [...]}}
            allowed = set(where["type_source"]["$in"])
            entries = [e for e in entries if e[1].get("type_source") in allowed]
        top = entries[:n_results]
        return {
            "documents": [[e[0] for e in top]],
            "metadatas": [[e[1] for e in top]],
            "distances": [[e[2] for e in top]],
        }


class FakeLLM:
    def __init__(self, answer="Réponse (GATS_2023.pdf, page 14)", fail=False):
        self.answer = answer
        self.fail = fail
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.fail:
            raise RuntimeError("clé invalide")
        return self.answer


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def tabac_collection():
    return FakeCollection([
        ("La prévalence est de 4,4%.", GATS_META | {"page": 14}, 0.30),
        ("Chez les 15-24 ans : 1,6%.", GATS_META | {"page": 49}, 0.50),
        ("Tendances mondiales.", OMS_META | {"page": 3}, 0.60),
        ("Texte lointain.", OMS_META | {"page": 9}, 1.20),
    ])
