"""
Prompt de génération (unique : utilisé par l'application ET l'évaluation)
et gestion des citations document + page.
"""

import config
from rag.sources import full_reference

SYSTEM_PROMPT = (
    "Tu es un assistant documentaire en santé publique pour le Sénégal (tabac, cancer du sein, "
    "cancer du col de l'utérus, paludisme), destiné à des chercheurs, des étudiants et des "
    "organisations. Tu réponds uniquement à partir des extraits de documents officiels fournis "
    "et tu cites le document et la page de chaque affirmation. Tu ne poses jamais de diagnostic "
    "et tu ne donnes jamais de conseil de traitement."
)

NO_ANSWER_SENTENCE = "Je ne trouve pas cette information dans les documents indexés."

USER_PROMPT_TEMPLATE = """Tu es un expert en santé publique au Sénégal. Ton public : chercheurs, étudiants, ONG et institutions.

RÈGLES DE SANTÉ (prioritaires sur tout le reste) :
- Tu informes à partir des sources. Tu ne poses JAMAIS de diagnostic : ne dis jamais à une personne qu'elle a, ou n'a pas, une maladie.
- Tu ne donnes JAMAIS de conseil de traitement personnel (médicament, dose, arrêt ou changement de traitement, choix thérapeutique).
- Si la question évoque des symptômes ou une situation médicale personnelle, rappelle que seul un professionnel
  de santé peut l'évaluer, puis ne donne que les informations générales présentes dans le contexte.

CONSIGNE :
- Réponds UNIQUEMENT à partir des extraits du CONTEXTE ci-dessous. Chaque extrait est étiqueté avec son document et sa page.
- Sois précis et factuel : donne les chiffres exacts avec leur unité, l'année de la mesure et la population concernée
  (ex : « adultes de 15 ans et plus », « femmes de 30 à 49 ans »).
- Fais suivre CHAQUE affirmation de sa source entre parenthèses, au format (titre du document, page n),
  en reprenant le titre tel qu'il apparaît dans l'étiquette de l'extrait.
- Le contexte peut contenir plusieurs chiffres proches pour des sous-groupes différents (global, par sexe, par âge,
  quotidien/occasionnel...). Ce n'est PAS une raison de refuser : donne le chiffre qui répond le plus directement
  à la question (par défaut l'ensemble de la population), puis signale les variantes utiles.
- Certains documents sont en anglais : réponds toujours en français, en traduisant fidèlement.
- Si les extraits ne permettent pas de répondre, écris exactement : « """ + NO_ANSWER_SENTENCE + """ »
  et, si c'est utile, précise ce que les documents contiennent de plus proche. Ne devine jamais.
  N'invente jamais de chiffre, de page ou de document.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE (chaque affirmation suivie de sa source entre parenthèses) :
"""


def build_context(hits):
    """Étiquette chaque extrait avec son document et sa page."""
    return "\n\n".join(
        f"[Extrait {i} — {h.get('titre') or h['source']}, page {h['page']}]\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def build_prompt(question, hits):
    return USER_PROMPT_TEMPLATE.format(context=build_context(hits), question=question)


def generate_answer(question, hits, llm):
    """Génère la réponse avec le modèle de langage configuré (voir llm/)."""
    return llm.complete(SYSTEM_PROMPT, build_prompt(question, hits))


def unique_citations(hits):
    """
    Une citation par (document, page), sans doublon, dans l'ordre de pertinence :
    {source, page, titre, theme, reference}.
    """
    citations, seen = [], set()
    for h in hits:
        key = (h["source"], h["page"])
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "source": h["source"],
            "page": h["page"],
            "titre": h.get("titre") or h["source"],
            "theme": h.get("theme"),
            "reference": full_reference(h, h["page"]),
        })
    return citations


def looks_like_refusal(answer_text):
    """Vrai si le modèle indique ne pas avoir trouvé l'information."""
    lowered = (answer_text or "").lower()
    return any(marker in lowered for marker in config.REFUSAL_MARKERS)
