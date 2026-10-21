"""Extraction du texte des PDF, page par page (pour citer un numéro de page précis)."""


def extract_pages(pdf_path):
    """Renvoie une liste de (numéro_de_page, texte), en ignorant les pages vides."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        content = page.extract_text()
        if content and content.strip():
            pages.append((page_number, content))
    return pages
