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