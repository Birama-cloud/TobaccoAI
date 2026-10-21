from ui.formatting import (
    answer_html, answer_meta_html, answer_text_html, brand_html, escape_text, extract_html,
    question_html, source_pills_html, theme_badge_html, warning_html,
)

MALICIOUS = '<img src=x onerror="alert(1)"><script>alert(2)</script>'


def test_escape_text():
    assert escape_text("<b>gras</b> & co") == "&lt;b&gt;gras&lt;/b&gt; &amp; co"
    assert escape_text("ligne 1\nligne 2") == "ligne 1<br>ligne 2"


def test_question_is_escaped_inside_html_block():
    rendered = question_html(MALICIOUS + "\n![image](http://exemple.org/x.png)")
    assert rendered.startswith('<div class="question-text">')
    assert "<script>" not in rendered and "<img" not in rendered
    assert "&lt;script&gt;" in rendered


def test_answer_is_escaped_but_keeps_bold_and_lists():
    rendered = answer_html(MALICIOUS + "\n- **4,4 %** des adultes")
    assert "<script>" not in rendered and "<img" not in rendered
    assert "• <strong>4,4 %</strong> des adultes" in rendered


def test_bold_cannot_smuggle_html():
    assert "<script>" not in answer_text_html("**<script>x</script>**")


def test_source_pills_escape_titles():
    rendered = source_pills_html([{"titre": "<b>Rapport</b>", "page": 12, "theme": "tabac"}])
    assert "&lt;b&gt;Rapport&lt;/b&gt;, page 12" in rendered
    assert "<b>" not in rendered


def test_extract_shows_exact_text_and_similarity():
    hit = {"text": "Texte <exact>", "titre": "GATS", "source": "g.pdf", "page": 49, "theme": "tabac", "distance": 0.14}
    rendered = extract_html(hit, 1)
    assert "Texte &lt;exact&gt;" in rendered
    assert "similarité 0,86" in rendered
    assert "GATS, page 49" in rendered


def test_answer_meta_counts_extracts_and_shows_theme_badges():
    rendered = answer_meta_html(["cancer_col"], 3)
    assert "Réponse fondée sur 3 extraits" in rendered
    assert 'class="theme-badge theme-cancer_col">Cancer du col<' in rendered
    assert "1 extrait<" in answer_meta_html(["tabac"], 1)
    assert "Aucune réponse" in answer_meta_html(["tabac"], 2, refused=True)


def test_other_blocks_escape_text():
    assert "<script>" not in warning_html(MALICIOUS)
    assert "<script>" not in brand_html(MALICIOUS)
    assert "theme-paludisme" in theme_badge_html("paludisme")
