from pathlib import Path

from .parser import parse_pdf
from .cleaner import (
    normalize_element,
    find_repeated_texts,
    is_repeated_header_or_footer,
)


def process_pdf(
    file_path: Path,
    multimodal: bool = False,
):
    """
    Parse and clean a PDF into normalized DocumentElements.
    """

    # ---------------------------------------------------------
    # Step 1: Parse PDF
    # ---------------------------------------------------------

    elements = parse_pdf(
        file_path=file_path,
        multimodal=multimodal,
    )

    # ---------------------------------------------------------
    # Step 2: Find repeated headers and footers
    # ---------------------------------------------------------

    repeated_texts = find_repeated_texts(
        elements,
        minimum_occurrences=3,
    )

    # ---------------------------------------------------------
    # Step 3: Normalize elements
    # ---------------------------------------------------------

    cleaned_elements = []

    document_id = file_path.stem
    document_name = file_path.name

    for element in elements:

        # Remove repeated headers/footers
        if is_repeated_header_or_footer(
            element,
            repeated_texts,
        ):
            continue

        normalized = normalize_element(
            element=element,
            document_id=document_id,
            document_name=document_name,
        )

        if normalized is not None:
            cleaned_elements.append(normalized)

    return cleaned_elements