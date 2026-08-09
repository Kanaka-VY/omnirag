from src.retrieval.models import RetrievedChunk


def build_context(
    chunks: list[RetrievedChunk],
) -> str:
    if not chunks:
        return ""

    sections = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        source = (
            f"{chunk.document_name}, "
            f"page(s) {chunk.page_numbers}"
        )

        section = f"""
[Source {index}]
Document: {chunk.document_name}
Pages: {chunk.page_numbers}
Section: {chunk.section}
Source ID: {chunk.chunk_id}

Content:
{chunk.text}
"""

        sections.append(section)

    return "\n".join(sections)