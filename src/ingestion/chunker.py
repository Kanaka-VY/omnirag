from typing import List
import hashlib
import uuid

from .chunk_schema import DocumentChunk
from .schema import DocumentElement


def classify_element(element_type: str) -> str:
    """
    Map Unstructured element types to a simplified
    multimodal content type.
    """

    if element_type == "Table":
        return "table"

    if element_type == "Image":
        return "image"

    if element_type in {
        "Title",
        "Header",
        "Footer",
        "NarrativeText",
        "ListItem",
        "Text",
    }:
        return "text"

    return "text"


def create_chunk_id(
    document_id: str,
    element_ids: list[str],
) -> str:
    """
    Create a deterministic UUID for a chunk.

    The same document + element IDs will always
    generate the same UUID.
    """

    source = f"{document_id}:{'|'.join(element_ids)}"

    digest = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()

    return str(uuid.UUID(digest[:32]))


def _build_chunk(
    elements: List[DocumentElement],
) -> DocumentChunk:
    """
    Combine multiple DocumentElements into one DocumentChunk.
    """

    document_id = elements[0].document_id
    document_name = elements[0].document_name

    text_parts = []

    page_numbers = []
    element_ids = []
    element_types = []

    contains_table = False
    contains_image = False

    section = elements[0].section

    for element in elements:

        if element.text:
            text_parts.append(element.text)

        if element.page_number is not None:
            page_numbers.append(element.page_number)

        element_ids.append(element.element_id)
        element_types.append(element.element_type)

        if element.element_type == "Table":
            contains_table = True

        if element.element_type == "Image":
            contains_image = True

    # ---------------------------------------------------------
    # Build chunk text
    # ---------------------------------------------------------

    text = "\n\n".join(text_parts)

    # ---------------------------------------------------------
    # Determine multimodal content type
    # ---------------------------------------------------------

    content_types = [
        classify_element(element.element_type)
        for element in elements
    ]

    if "table" in content_types:
        content_type = "table"
    elif "image" in content_types:
        content_type = "image"
    else:
        content_type = "text"

    # ---------------------------------------------------------
    # Preserve table structure
    # ---------------------------------------------------------

    table_data = (
        text
        if content_type == "table"
        else None
    )

    # ---------------------------------------------------------
    # Create deterministic chunk ID
    # ---------------------------------------------------------

    chunk_id = create_chunk_id(
        document_id=document_id,
        element_ids=element_ids,
    )

    # ---------------------------------------------------------
    # Create DocumentChunk
    # ---------------------------------------------------------

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        text=text,
        section=section,
        page_numbers=sorted(set(page_numbers)),
        element_ids=element_ids,
        element_types=element_types,
        contains_table=contains_table,
        contains_image=contains_image,
        content_type=content_type,
        table_data=table_data,
    )


def create_chunks(
    elements: List[DocumentElement],
    max_characters: int = 1000,
) -> List[DocumentChunk]:
    """
    Convert DocumentElements into semantically meaningful chunks.

    Rules:
    1. A Title starts a new section.
    2. Tables are standalone chunks.
    3. Chunks respect the maximum character limit.
    4. Section metadata is preserved.
    """

    chunks = []

    current_elements: List[DocumentElement] = []
    current_length = 0

    current_section = None

    for element in elements:

        element_type = element.element_type

        # -----------------------------------------------------
        # TITLE → start a new section
        # -----------------------------------------------------

        if element_type == "Title":

            if current_elements:
                chunks.append(
                    _build_chunk(current_elements)
                )

                current_elements = []
                current_length = 0

            current_section = element.text
            element.section = current_section

        else:
            element.section = current_section

        # -----------------------------------------------------
        # TABLE → always standalone
        # -----------------------------------------------------

        if element_type == "Table":

            if current_elements:
                chunks.append(
                    _build_chunk(current_elements)
                )

                current_elements = []
                current_length = 0

            chunks.append(
                _build_chunk([element])
            )

            continue

        # -----------------------------------------------------
        # CHARACTER LIMIT
        # -----------------------------------------------------

        element_length = len(
            element.text or ""
        )

        if (
            current_elements
            and current_length + element_length
            > max_characters
        ):

            chunks.append(
                _build_chunk(current_elements)
            )

            current_elements = []
            current_length = 0

        current_elements.append(element)
        current_length += element_length

    # ---------------------------------------------------------
    # FINAL CHUNK
    # ---------------------------------------------------------

    if current_elements:
        chunks.append(
            _build_chunk(current_elements)
        )

    return chunks


def split_text(
    text: str,
    max_characters: int,
) -> list[str]:
    """
    Split oversized text into approximately equal
    character-based pieces.
    """

    return [
        text[i:i + max_characters]
        for i in range(
            0,
            len(text),
            max_characters,
        )
    ]


def _split_oversized_element(
    element: DocumentElement,
    max_characters: int,
) -> List[DocumentChunk]:
    """
    Split a single oversized element into multiple chunks.
    """

    chunks = []

    parts = split_text(
        element.text,
        max_characters,
    )

    content_type = classify_element(
        element.element_type
    )

    for index, part in enumerate(parts):

        chunk_id = create_chunk_id(
            element.document_id,
            [
                f"{element.element_id}:{index}"
            ],
        )

        table_data = (
            part
            if content_type == "table"
            else None
        )

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=element.document_id,
                document_name=element.document_name,
                text=part,
                section=element.section,
                page_numbers=(
                    [element.page_number]
                    if element.page_number is not None
                    else []
                ),
                element_ids=[
                    element.element_id
                ],
                element_types=[
                    element.element_type
                ],
                contains_table=(
                    element.element_type == "Table"
                ),
                contains_image=(
                    element.element_type == "Image"
                ),
                content_type=content_type,
                table_data=table_data,
            )
        )

    return chunks