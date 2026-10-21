import json
import re

import pytest

import config
from llm import create_llm_client


def test_defaults_use_openai():
    settings = config.load_settings(env={})
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_api_key is None
    assert settings.needs_api_key


def test_env_values_override_defaults():
    settings = config.load_settings(env={"LLM_PROVIDER": "OpenAI ", "LLM_MODEL": "autre-modele", "OPENAI_API_KEY": "cle-test"})
    assert settings.llm_model == "autre-modele"
    assert settings.llm_api_key == "cle-test"


def test_ollama_needs_no_key_and_uses_local_address():
    settings = config.load_settings(env={"LLM_PROVIDER": "ollama", "LLM_MODEL": "mon-modele"})
    assert not settings.needs_api_key
    assert settings.llm_base_url.startswith("http://localhost")


def test_gemini_reads_its_own_key():
    settings = config.load_settings(env={"LLM_PROVIDER": "gemini", "LLM_MODEL": "m", "GEMINI_API_KEY": "g", "OPENAI_API_KEY": "o"})
    assert settings.llm_api_key == "g"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        config.load_settings(env={"LLM_PROVIDER": "inconnu"})


def test_model_required_when_no_default():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        config.load_settings(env={"LLM_PROVIDER": "gemini"})


def test_llm_client_requires_key_for_openai():
    with pytest.raises(ValueError):
        create_llm_client(config.load_settings(env={}))


def test_llm_client_typed_key_wins_over_env_key():
    settings = config.load_settings(env={"OPENAI_API_KEY": "cle-env"})
    client = create_llm_client(settings, api_key="cle-saisie")
    assert client._client.api_key == "cle-saisie"


def test_four_themes_with_their_folders_and_colors():
    assert list(config.THEMES) == ["tabac", "cancer_sein", "cancer_col", "paludisme"]
    collections = [t["collection"] for t in config.THEMES.values()]
    assert len(set(collections)) == 4
    assert config.THEMES["paludisme"]["couleur"] == "#4338CA"
    for theme in config.THEMES:
        assert config.theme_docs_dir(theme).is_dir()
    with pytest.raises(ValueError):
        config.theme_docs_dir("../autre")


def test_tabac_eval_set_is_valid():
    items = json.loads((config.PROJECT_ROOT / config.THEMES["tabac"]["eval_file"]).read_text(encoding="utf-8"))
    assert len({i["id"] for i in items}) == len(items) >= 18
    for item in items:
        if item["expects_answer"]:
            assert item["expected_keywords"] and item["expected_pages"]


def test_env_file_is_git_ignored():
    lines = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines


def test_no_api_key_written_in_code():
    pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
    for path in config.PROJECT_ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_platform_name_and_notice():
    assert config.NOM_PLATEFORME
    assert "Vos questions ne sont pas enregistrées" in config.IMPORTANT_NOTICE
    assert "aucune donnée permettant d'identifier une personne" in config.IMPORTANT_NOTICE


def test_suggestions_only_for_themes_with_documents():
    for theme, questions in config.SUGGESTIONS.items():
        assert len(questions) <= 3
        if questions:
            assert list(config.theme_docs_dir(theme).glob("*.pdf")), f"suggestions sans document : {theme}"
