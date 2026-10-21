import evaluate_rag
from rag import pipeline
from rag.generation import NO_ANSWER_SENTENCE, SYSTEM_PROMPT, build_context, build_prompt, looks_like_refusal, unique_citations
from rag.pipeline import answer_question
from tests.conftest import FakeLLM

HITS = [
    {"text": "A", "source": "GATS_2023.pdf", "titre": "GATS Sénégal 2023", "titre_complet": "Enquête GATS",
     "producteur": "ANSD", "annee": 2023, "theme": "tabac", "page": 14, "distance": 0.3},
    {"text": "B", "source": "GATS_2023.pdf", "titre": "GATS Sénégal 2023", "titre_complet": "Enquête GATS",
     "producteur": "ANSD", "annee": 2023, "theme": "tabac", "page": 14, "distance": 0.4},
    {"text": "C", "source": "OMS.pdf", "page": 7, "theme": "tabac", "distance": 0.5},
]


def test_unique_citations_keeps_order_without_duplicates():
    citations = unique_citations(HITS)
    assert [(c["source"], c["page"]) for c in citations] == [("GATS_2023.pdf", 14), ("OMS.pdf", 7)]
    assert citations[0]["titre"] == "GATS Sénégal 2023"
    assert citations[0]["reference"] == "ANSD. Enquête GATS. 2023, p. 14."
    assert citations[1]["titre"] == "OMS.pdf"  # document sans titre : nom du fichier


def test_context_labels_each_extract_with_title_and_page():
    context = build_context(HITS)
    assert "[Extrait 1 — GATS Sénégal 2023, page 14]\nA" in context
    assert "[Extrait 3 — OMS.pdf, page 7]\nC" in context


def test_prompt_contains_health_rules():
    prompt = build_prompt("J'ai une boule au sein, est-ce grave ?", HITS)
    assert "JAMAIS de diagnostic" in prompt
    assert "JAMAIS de conseil de traitement" in prompt
    assert "professionnel" in prompt
    assert "jamais de diagnostic" in SYSTEM_PROMPT


def test_prompt_targets_researchers_and_requires_sources():
    prompt = build_prompt("question", HITS)
    assert "chercheurs" in prompt and "étudiants" in prompt
    assert "l'année de la mesure et la population concernée" in prompt
    assert "CHAQUE affirmation" in prompt
    assert NO_ANSWER_SENTENCE in prompt and looks_like_refusal(NO_ANSWER_SENTENCE)


def test_prompt_keeps_question_braces_intact():
    assert "{x}" in build_prompt("Que vaut {x} ?", HITS)


def test_looks_like_refusal():
    assert looks_like_refusal("Je ne trouve pas cette information dans les documents.")
    assert not looks_like_refusal("La prévalence est de 4,4 %.")
    assert not looks_like_refusal(None)


def test_pipeline_guardrail_does_not_call_llm(embedder, tabac_collection):
    llm = FakeLLM()
    result = answer_question("recette de cuisine", embedder, {"tabac": tabac_collection}, llm, threshold=0.1)
    assert result.guardrail_triggered and result.refused
    assert result.answer is None
    assert llm.calls == []


def test_pipeline_sends_only_relevant_hits_with_shared_prompt(embedder, tabac_collection):
    llm = FakeLLM()
    result = answer_question("taux de tabagisme", embedder, {"tabac": tabac_collection}, llm)

    assert result.answer == llm.answer and not result.refused
    assert [(c["source"], c["page"]) for c in result.citations] == [
        ("GATS_2023.pdf", 14), ("GATS_2023.pdf", 49), ("OMS.pdf", 3)]
    assert result.themes == ["tabac"]
    system_prompt, user_prompt = llm.calls[0]
    assert system_prompt == SYSTEM_PROMPT
    assert user_prompt == build_prompt("taux de tabagisme", result.relevant_hits)
    assert "Texte lointain." not in user_prompt   # morceau au-delà du seuil exclu


def test_pipeline_captures_llm_errors(embedder, tabac_collection):
    result = answer_question("taux", embedder, {"tabac": tabac_collection}, FakeLLM(fail=True))
    assert result.answer is None
    assert "clé invalide" in result.error


def test_evaluation_uses_the_same_pipeline_as_the_app():
    assert evaluate_rag.answer_question is pipeline.answer_question


def test_evaluation_checks_expected_document_and_page():
    hits = [{"source": "OMS.pdf", "page": 94}]
    assert evaluate_rag.page_retrieved({"expected_pages": [94], "expected_source": "GATS_2023.pdf"}, hits) is False
    assert evaluate_rag.page_retrieved({"expected_pages": [94]}, hits) is True
    assert evaluate_rag.page_retrieved({"expected_pages": []}, hits) is None
