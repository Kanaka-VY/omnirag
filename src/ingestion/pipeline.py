from pathlib import Path

from .cleaner import (
    find_repeated_texts,
    is_repeated_header_or_footer,
    normalize_element,
)
from .parser import parse_pdf


def process_pdf(file_path: Path):
    elements = parse_pdf(file_path)

    repeated_texts = find_repeated_texts(elements)

    document_id = file_path.stem
    document_name = file_path.name

    normalized_elements = []
    current_section = None

    for element in elements:
        if is_repeated_header_or_footer(
            element,
            repeated_texts,
        ):
            continue

        element_type = type(element).__name__

        normalized = normalize_element(
            element=element,
            document_id=document_id,
            document_name=document_name,
            section=current_section,
        )

        if normalized is None:
            continue

        normalized_elements.append(normalized)

        if element_type == "Title":
            current_section = normalized.text

    return normalized_elements