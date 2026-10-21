"""
Construction des blocs HTML affichés avec unsafe_allow_html=True.

Tout texte qui ne vient pas de nous (question, réponse du modèle, titre ou nom
de fichier, extrait de PDF) est échappé avec html.escape : du HTML saisi dans
une question s'affiche comme du texte, au lieu d'être interprété.
"""

import html
import re

import config

SHIELD_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 3l7 3v6c0 4.5-3 7.6-7 9-4-1.4-7-4.5-7-9V6l7-3z"/><circle cx="12" cy="11" r="2.2"/></svg>'
)
DOC_ICON = (
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'aria-hidden="true"><path d="M14 3H6v18h12V7l-4-4z"/><path d="M14 3v4h4"/></svg>'
)


def escape_text(text):
    """Échappe le HTML et conserve les retours à la ligne."""
    return html.escape(str(text)).replace("\n", "<br>")


def answer_text_html(text):
    """
    Réponse du modèle : texte échappé, puis mise en forme minimale et sûre
    (**gras** et listes à tirets), appliquée APRÈS l'échappement.
    """
    lines = []
    for line in html.escape(str(text)).split("\n"):
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            line = "• " + stripped[2:]
        lines.append(line)
    escaped = "<br>".join(lines)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def question_html(question):
    """Question de l'utilisateur, dans un bloc HTML : ni HTML ni Markdown n'y sont interprétés."""
    return f'<div class="question-text">{escape_text(question)}</div>'


def brand_html(name=config.NOM_PLATEFORME, badge=None):
    badge_html = f'<span class="brand-badge">{escape_text(badge)}</span>' if badge else ""
    return (
        f'<div class="brand"><div class="brand-logo">{SHIELD_ICON}</div>'
        f'<div class="brand-name">{escape_text(name)}</div>{badge_html}</div>'
    )


def theme_badge_html(theme):
    label = config.THEMES[theme]["label_court"] if theme in config.THEMES else str(theme)
    return f'<span class="theme-badge theme-{html.escape(str(theme))}">{escape_text(label)}</span>'


def answer_meta_html(themes, n_extracts, refused=False):
    badges = "".join(theme_badge_html(t) for t in themes)
    pluriel = "s" if n_extracts > 1 else ""
    if refused:
        count = f"Aucune réponse dans les {n_extracts} extrait{pluriel} consulté{pluriel}"
    else:
        count = f"Réponse fondée sur {n_extracts} extrait{pluriel}"
    return f'<div class="answer-meta">{badges}<span class="answer-count">{count}</span></div>'


def answer_html(answer):
    return f'<div class="answer-text">{answer_text_html(answer)}</div>'


def answer_note_html():
    return (
        '<div class="answer-note">Réponse limitée aux documents indexés. '
        "Vérifiez toujours les extraits avant de citer.</div>"
    )


def source_pills_html(citations):
    """citations : liste de dicts {titre, page, theme}."""
    pills = "".join(
        f'<span class="source-pill theme-{html.escape(str(c.get("theme", "")))}">{DOC_ICON}'
        f'{escape_text(c["titre"])}, page {escape_text(c["page"])}</span>'
        for c in citations
    )
    return f'<div class="source-pills">{pills}</div>'


def extract_html(hit, index):
    similarity = f"{1 - hit['distance']:.2f}".replace(".", ",")
    return (
        f'<div class="extract theme-{html.escape(str(hit.get("theme", "")))}">'
        f'<div class="extract-head">Extrait {index} — {escape_text(hit.get("titre") or hit["source"])}, '
        f'page {escape_text(hit["page"])} <span class="extract-score">· similarité {similarity}</span></div>'
        f'<div class="extract-text">{html.escape(str(hit["text"]))}</div></div>'
    )


def warning_html(text):
    return f'<div class="answer-warning">{escape_text(text)}</div>'


def important_notice_html(text=config.IMPORTANT_NOTICE):
    return f'<div class="notice-important"><strong>Important</strong>{escape_text(text)}</div>'


def coming_soon_html(title, description):
    return (
        f'<div class="coming-soon"><span class="tag">Bientôt disponible</span>'
        f"<h2>{escape_text(title)}</h2><p>{escape_text(description)}</p></div>"
    )
