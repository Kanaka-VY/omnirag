import re
from collections import Counter
from typing import Optional
from .schema import DocumentElement
from .schema import DocumentElement

def clean_text(text: Optional[str]) -> str:
    """
    Normalize whitespace in extracted text.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def should_remove_element(element_type: str, text: str) -> bool:
    """
    Decide whether an element should be removed
    from semantic content.
    """
    if not text.strip():
        return True

    if element_type == "PageNumber":
        return True

    return False


def find_repeated_texts(
    elements,
    minimum_occurrences: int = 3,
) -> set[str]:
    """
    Find text values that occur repeatedly.
    """
    texts = []

    for element in elements:
        text = clean_text(str(element))

        if text:
            texts.append(text)

    counts = Counter(texts)

    return {
        text
        for text, count in counts.items()
        if count >= minimum_occurrences
    }


def is_repeated_header_or_footer(
    element,
    repeated_texts: set[str],
) -> bool:
    element_type = type(element).__name__
    text = clean_text(str(element))

    return (
        element_type in {"Header", "Footer"}
        and text in repeated_texts
    )

def normalize_element(
    element,
    document_id: str,
    document_name: str,
    section: Optional[str] = None,
) -> Optional[DocumentElement]:
    element_type = type(element).__name__
    text = clean_text(str(element))

    if should_remove_element(element_type, text):
        return None

    metadata = element.metadata

    return DocumentElement(
        element_id=getattr(element, "id", ""),
        document_id=document_id,
        document_name=document_name,
        element_type=element_type,
        text=text,
        page_number=getattr(metadata, "page_number", None),
        parent_id=getattr(metadata, "parent_id", None),
        section=section,
        text_as_html=getattr(metadata, "text_as_html", None),
    )