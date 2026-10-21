"""
Client unique du modèle de langage.

OpenAI, Gemini et Ollama exposent tous une API compatible OpenAI : on utilise
donc le même SDK et on change seulement l'adresse du service (base_url) et le
nom du modèle, choisis dans config.py / .env. Le reste du code n'appelle que
LLMClient.complete() et ne sait pas quel fournisseur est utilisé.
"""

import config


class LLMClient:
    def __init__(self, provider, model, api_key=None, base_url=None, temperature=config.LLM_TEMPERATURE):
        from openai import OpenAI

        self.provider = provider
        self.model = model
        self.temperature = temperature
        # Ollama n'a pas de clé, mais le SDK en exige une non vide.
        self._client = OpenAI(api_key=api_key or "sans-cle", base_url=base_url)

    def complete(self, system_prompt, user_prompt):
        """Envoie un message au modèle et renvoie le texte de sa réponse."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


def create_llm_client(settings, api_key=None):
    """
    Construit le client à partir des réglages. `api_key` (saisie dans
    l'interface ou en ligne de commande) est prioritaire sur la clé du .env.
    """
    key = api_key or settings.llm_api_key
    if settings.needs_api_key and not key:
        raise ValueError(f"Aucune clé d'API pour {settings.provider_label}.")
    return LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=key,
        base_url=settings.llm_base_url,
    )
