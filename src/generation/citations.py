from src.retrieval.models import RetrievedChunk


def format_sources(
    chunks: list[RetrievedChunk],
) -> list[str]:

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
    context_items,
):
    citations = []

    for item in context_items:

        metadata = item.metadata

        citations.append(
            {
                "chunk_id": item.chunk_id,
                "document_id": metadata.get(
                    "document_id"
                ),
                "page_numbers": metadata.get(
                    "page_numbers",
                    [],
                ),
                "content_type": metadata.get(
                    "content_type"
                ),
            }
        )

    return citations