"""
Pipeline complet question -> réponse.

C'est LA fonction appelée par l'application Streamlit ET par evaluate_rag.py :
même recherche, même seuil, même prompt, même modèle. L'évaluation mesure donc
exactement ce que voit l'utilisateur.
"""

from dataclasses import dataclass, field

import config
from rag.generation import generate_answer, looks_like_refusal, unique_citations
from rag.retrieval import filter_relevant, search


@dataclass
class RAGResult:
    question: str
    hits: list = field(default_factory=list)            # tous les morceaux trouvés
    relevant_hits: list = field(default_factory=list)   # ceux qui passent le seuil
    answer: str | None = None
    error: str | None = None

    @property
    def best_distance(self):
        return self.hits[0]["distance"] if self.hits else None

    @property
    def guardrail_triggered(self):
        """Vrai si aucun morceau ne passe le seuil : aucune réponse n'est générée."""
        return not self.relevant_hits

    @property
    def refused(self):
        return self.guardrail_triggered or (self.answer is not None and looks_like_refusal(self.answer))

    @property
    def citations(self):
        return unique_citations(self.relevant_hits)

    @property
    def themes(self):
        """Thèmes des extraits utilisés, dans l'ordre de pertinence."""
        return list(dict.fromkeys(h["theme"] for h in self.relevant_hits))


def answer_question(question, embedder, collections, llm,
                    top_k=config.TOP_K, threshold=config.RELEVANCE_DISTANCE_THRESHOLD, source_types=None):
    """
    1. Recherche dans les collections des thèmes choisis, limitée aux types de
       source cochés (`source_types`, None = tous).
    2. Garde-fou : si rien ne passe le seuil de pertinence, on ne génère rien.
    3. Génération à partir des seuls morceaux pertinents.
    La question n'est jamais enregistrée sur le disque.
    """
    result = RAGResult(question=question)
    result.hits = search(question, embedder, collections, top_k=top_k, source_types=source_types)
    result.relevant_hits = filter_relevant(result.hits, threshold)

    if result.guardrail_triggered:
        return result

    try:
        result.answer = generate_answer(question, result.relevant_hits, llm)
    except Exception as e:
        result.error = str(e)
    return result
