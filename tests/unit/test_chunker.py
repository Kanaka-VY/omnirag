

from src.ingestion.chunk_schema import DocumentChunk
from src.ingestion.chunker import create_chunks
from src.ingestion.schema import DocumentElement


def make_element(
    element_id: str,
    element_type: str,
    text: str,
    section: str | None = None,
    page_number: int = 1,
) -> DocumentElement:
    return DocumentElement(
        element_id=element_id,
        document_id="doc1",
        document_name="sample.pdf",
        element_type=element_type,
        text=text,
        page_number=page_number,
        section=section,
    )

def test_title_creates_new_section():
    elements = [
        make_element(
            "1",
            "Title",
            "Leave Policy",
        ),
        make_element(
            "2",
            "NarrativeText",
            "Employees receive annual leave.",
        ),
        make_element(
            "3",
            "Title",
            "Benefits",
        ),
        make_element(
            "4",
            "NarrativeText",
            "Employees receive health insurance.",
        ),
    ]

    chunks = create_chunks(
        elements,
        max_characters=1000,
    )

    assert len(chunks) == 2
    assert chunks[0].section == "Leave Policy"
    assert chunks[1].section == "Benefits"

def test_table_is_standalone_chunk():
    elements = [
        make_element(
            "1",
            "NarrativeText",
            "Revenue increased.",
        ),
        make_element(
            "2",
            "Table",
            "Year Revenue\n2024 $100M\n2025 $130M",
        ),
        make_element(
            "3",
            "NarrativeText",
            "Growth was driven by sales.",
        ),
    ]

    chunks = create_chunks(
        elements,
        max_characters=1000,
    )

    assert len(chunks) == 3

    assert chunks[1].contains_table is True
    assert chunks[1].element_types == ["Table"]

def test_chunks_respect_character_limit():
    elements = [
        make_element(
            "1",
            "NarrativeText",
            "A" * 100,
        ),
        make_element(
            "2",
            "NarrativeText",
            "B" * 100,
        ),
        make_element(
            "3",
            "NarrativeText",
            "C" * 100,
        ),
    ]

    chunks = create_chunks(
        elements,
        max_characters=150,
    )

    for chunk in chunks:
        assert len(chunk.text) <= 150

def test_chunk_preserves_metadata():
    elements = [
        make_element(
            "1",
            "NarrativeText",
            "First paragraph.",
            section="Introduction",
            page_number=2,
        ),
        make_element(
            "2",
            "NarrativeText",
            "Second paragraph.",
            section="Introduction",
            page_number=3,
        ),
    ]

    chunks = create_chunks(
        elements,
        max_characters=1000,
    )

    assert chunks[0].document_id == "doc1"
    assert chunks[0].document_name == "sample.pdf"
    assert chunks[0].page_numbers == [2, 3]
    assert chunks[0].element_ids == ["1", "2"]