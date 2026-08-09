from types import SimpleNamespace

from src.generation.context import ContextBuilder


def test_context_builder_deduplicates_chunks():

    results = [
        SimpleNamespace(
            chunk_id="A",
            text="Ravi salary is 60000.",
            score=0.9,
            metadata={
                "document_id": "employees.pdf",
                "page_numbers": [3],
                "content_type": "table",
            },
        ),
        SimpleNamespace(
            chunk_id="A",
            text="Ravi salary is 60000.",
            score=0.8,
            metadata={
                "document_id": "employees.pdf",
                "page_numbers": [3],
                "content_type": "table",
            },
        ),
        SimpleNamespace(
            chunk_id="B",
            text="Ravi works in AI.",
            score=0.7,
            metadata={
                "document_id": "employees.pdf",
                "page_numbers": [3],
                "content_type": "text",
            },
        ),
    ]

    builder = ContextBuilder(
        max_chunks=5
    )

    context, selected = builder.build(
        results
    )

    assert len(selected) == 2
    assert "Source 1" in context
    assert "Source 2" in context