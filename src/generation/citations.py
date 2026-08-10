from src.retrieval.models import RetrievedChunk


def format_sources(
    chunks: list[RetrievedChunk],
) -> list[str]:
    """
    Format retrieved chunks as human-readable sources.
    """

    sources = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        source = (
            f"[Source {index}] "
            f"{chunk.document_name}, "
            f"page(s) {chunk.page_numbers}"
        )

        sources.append(source)

    return sources


def build_citations(
    context_items: list[RetrievedChunk],
) -> list[dict]:
    """
    Build structured citation information
    from retrieved chunks.
    """

    citations = []

    for item in context_items:
        citations.append(
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "document_name": item.document_name,
                "page_numbers": item.page_numbers,
                "content_type": item.content_type,
            }
        )

    return citations