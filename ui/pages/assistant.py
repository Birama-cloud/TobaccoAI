"""
Page Assistant : questions-réponses sourcées sur le corpus documentaire.

L'historique de la conversation reste en mémoire de session (st.session_state) :
il disparaît à la fermeture de l'onglet et n'est jamais écrit sur le disque.
"""

import streamlit as st

import config
from llm import create_llm_client
from rag.pipeline import answer_question
from rag.store import get_collections, indexed_documents, list_pdf_files
from ui.formatting import (
    answer_html, answer_meta_html, answer_note_html, escape_text, extract_html,
    important_notice_html, question_html, source_pills_html, warning_html,
)
from ui.resources import get_chroma, get_embedder, get_settings, prepare_corpus

ALL = "tous"


# ---------------------------------------------------------------- état
def _state():
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("theme_choice", ALL)
    return st.session_state


def _toggle(flag):
    st.session_state[flag] = not st.session_state.get(flag, False)


def _new_conversation():
    for key in [k for k in st.session_state if str(k).startswith(("show-ext-", "show-ref-"))]:
        del st.session_state[key]
    st.session_state["chat"] = []


def _ask(question):
    st.session_state["pending_question"] = question


# ---------------------------------------------------------------- colonne de gauche
def _theme_panel(chunk_counts):
    state = _state()
    if state.theme_choice != ALL and chunk_counts.get(state.theme_choice, 0) == 0:
        state.theme_choice = ALL

    with st.container(key="panel-themes"):
        st.markdown('<p class="panel-title">Thème</p>', unsafe_allow_html=True)
        options = [(ALL, config.ALL_THEMES_LABEL)] + [(k, t["label"]) for k, t in config.THEMES.items()]
        for key, label in options:
            available = key == ALL or chunk_counts.get(key, 0) > 0
            if not available:
                label += " · à indexer" if list_pdf_files(key) else " · documents à venir"
            if st.button(label, key=f"theme-{key}", width="stretch", disabled=not available,
                         type="primary" if state.theme_choice == key else "secondary"):
                state.theme_choice = key
                st.rerun()

    if state.theme_choice == ALL:
        return [t for t in config.THEMES if chunk_counts.get(t, 0) > 0]
    return [state.theme_choice]


def _source_panel(documents):
    with st.container(key="panel-sources"):
        st.markdown('<p class="panel-title">Type de source</p>', unsafe_allow_html=True)
        checked = [key for key, label in config.SOURCE_TYPES.items()
                   if st.checkbox(label, value=True, key=f"src-{key}")]

        n = len(documents)
        st.markdown(f'<p class="panel-caption">{n} document{"s" if n > 1 else ""} indexé{"s" if n > 1 else ""}</p>',
                    unsafe_allow_html=True)
        if documents:
            with st.popover("voir la liste", type="tertiary"):
                for meta in documents:
                    st.markdown(
                        f"**{escape_text(meta.get('titre_complet') or meta.get('titre') or meta['source'])}**  \n"
                        f"{escape_text(meta.get('producteur') or 'Producteur non renseigné')} · "
                        f"{meta.get('annee') or 'année non renseignée'} · "
                        f"{config.THEMES.get(meta.get('theme'), {}).get('label', '')}"
                    )

    # Toutes les cases cochées = aucun filtre (les PDF absents du registre restent cherchables).
    return None if len(checked) == len(config.SOURCE_TYPES) else checked


def _api_key_panel(settings):
    """Champ de clé uniquement si aucune clé n'est configurée dans .env."""
    if not settings.needs_api_key or settings.llm_api_key:
        return settings.llm_api_key
    with st.container(key="panel-key"):
        st.markdown('<p class="panel-title">Clé d\'API</p>', unsafe_allow_html=True)
        return st.text_input(
            f"Clé API {settings.provider_label}", type="password", label_visibility="collapsed",
            placeholder=f"Clé {settings.provider_label}", help="Gardée dans votre session, jamais enregistrée.",
        ) or None


# ---------------------------------------------------------------- conversation
def _suggestions(themes):
    questions = [q for t in themes for q in config.SUGGESTIONS.get(t, [])]
    return questions[:3]


def _render_assistant_message(message, index):
    with st.chat_message("assistant"):
        if message["status"] != "ok":
            st.markdown(warning_html(message["text"]), unsafe_allow_html=True)
            return

        st.markdown(
            answer_meta_html(message["themes"], len(message["extracts"]), message["refused"])
            + answer_html(message["answer"])
            + answer_note_html(),
            unsafe_allow_html=True,
        )
        from ui.navigation import PAGES
        with st.container(key=f"data-link-{index}"):
            st.page_link(PAGES["espace"], label="Voir les données associées")
        st.markdown(source_pills_html(message["citations"]), unsafe_allow_html=True)

        ext_flag, ref_flag = f"show-ext-{index}", f"show-ref-{index}"
        with st.container(horizontal=True, key=f"actions-{index}"):
            st.button("Masquer les extraits" if st.session_state.get(ext_flag) else "Voir les extraits",
                      key=f"btn-ext-{index}", on_click=_toggle, args=(ext_flag,))
            st.button("Copier la référence", key=f"btn-ref-{index}", on_click=_toggle, args=(ref_flag,))

        if st.session_state.get(ref_flag):
            st.caption("Cliquez sur l'icône de copie en haut à droite du cadre.")
            st.code("\n".join(c["reference"] for c in message["citations"]), language=None, wrap_lines=True)
        if st.session_state.get(ext_flag):
            st.markdown("".join(extract_html(h, i) for i, h in enumerate(message["extracts"], start=1)),
                        unsafe_allow_html=True)


def _no_hit_text(themes, source_types):
    scope = ", ".join(config.THEMES[t]["label"] for t in themes) or "aucun thème"
    filtre = "" if source_types is None else " et les types de source cochés"
    return (
        f"Aucun extrait des documents indexés ({scope}{filtre}) n'est assez proche de votre question : "
        "je préfère ne pas répondre sans source. Essayez de reformuler votre question, de choisir "
        "« Tous les thèmes » ou de cocher d'autres types de source."
    )


def _answer(question, themes, source_types, settings, api_key):
    """Calcule la réponse et renvoie le message à ajouter à l'historique (en mémoire)."""
    if source_types is not None and not source_types:
        return {"role": "assistant", "status": "warning",
                "text": "Cochez au moins un type de source pour lancer la recherche."}
    if settings.needs_api_key and not api_key:
        return {"role": "assistant", "status": "warning",
                "text": f"Aucune clé d'API {settings.provider_label} : renseignez-la dans le fichier .env "
                        "ou dans le champ « Clé d'API » à gauche."}

    collections = get_collections(get_chroma(), themes)
    result = answer_question(question, get_embedder(), collections, create_llm_client(settings, api_key),
                             source_types=source_types)

    if result.guardrail_triggered:
        return {"role": "assistant", "status": "warning", "text": _no_hit_text(themes, source_types)}
    if result.error:
        return {"role": "assistant", "status": "warning",
                "text": f"Le modèle de langage n'a pas pu répondre ({result.error}). Réessayez dans un instant."}
    return {
        "role": "assistant",
        "status": "ok",
        "answer": result.answer,
        "refused": result.refused,
        "themes": result.themes,
        "citations": result.citations,
        "extracts": [
            {k: h.get(k) for k in ("text", "source", "titre", "page", "theme", "distance")}
            for h in result.relevant_hits
        ],
    }


# ---------------------------------------------------------------- page
def render():
    state = _state()
    settings = get_settings()
    warnings = prepare_corpus()

    client = get_chroma()
    all_collections = get_collections(client, config.THEMES)
    chunk_counts = {t: c.count() for t, c in all_collections.items()}

    left, main = st.columns([1, 2.8], gap="large")

    with left:
        themes = _theme_panel(chunk_counts)
        documents = [meta for t in themes for meta in indexed_documents(all_collections[t]).values()]
        source_types = _source_panel(documents)
        api_key = _api_key_panel(settings)
        st.markdown(important_notice_html(), unsafe_allow_html=True)
        for w in warnings:
            st.caption(w)

    with main:
        if not state.chat:
            st.markdown(
                '<div class="chat-empty">Posez une question sur le corpus : chaque réponse cite le document et la '
                "page de chaque information, et vous pouvez consulter les extraits utilisés.</div>",
                unsafe_allow_html=True,
            )

        for index, message in enumerate(state.chat):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(question_html(message["content"]), unsafe_allow_html=True)
            else:
                _render_assistant_message(message, index)

        suggestions = _suggestions(themes)
        if suggestions:
            with st.container(key="suggestions", horizontal=True):
                for j, q in enumerate(suggestions):
                    st.button(q, key=f"sugg-{j}", on_click=_ask, args=(q,))

        if state.chat:
            st.button("Nouvelle conversation", icon=":material/add_comment:", on_click=_new_conversation,
                      type="tertiary")

        typed = st.chat_input("Posez une autre question…" if state.chat else "Posez votre question…")
        question = typed or state.pop("pending_question", None)

        if question:
            if not themes:
                state.chat.append({"role": "user", "content": question})
                state.chat.append({"role": "assistant", "status": "warning",
                                   "text": "Aucun document n'est encore indexé : impossible de répondre sans source."})
                st.rerun()
            with st.chat_message("user"):
                st.markdown(question_html(question), unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Recherche dans les documents et rédaction de la réponse…"):
                    reply = _answer(question, themes, source_types, settings, api_key)
            state.chat.append({"role": "user", "content": question})
            state.chat.append(reply)
            st.rerun()
