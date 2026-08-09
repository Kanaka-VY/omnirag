from src.retrieval.models import RetrievedChunk


class ContextBuilder:
    """
    Builds grounded context from retrieved chunks.

    Responsibilities:
    - Remove duplicate chunks
    - Keep the highest-scoring duplicate
    - Sort by retrieval score
    - Limit the number of chunks
    - Format useful metadata for the LLM
    """

    def __init__(self, max_chunks: int = 5):
        self.max_chunks = max_chunks

    def _deduplicate(self, chunks) -> list:
        """
        Deduplicate chunks by chunk_id.

        If the same chunk appears more than once,
        keep the occurrence with the highest score.
        """

        unique_chunks = {}

        for chunk in chunks:
            chunk_id = str(
                getattr(chunk, "chunk_id", "")
            )

            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = chunk
                continue

            existing = unique_chunks[chunk_id]

            existing_score = float(
                getattr(existing, "score", 0.0)
            )

            current_score = float(
                getattr(chunk, "score", 0.0)
            )

            if current_score > existing_score:
                unique_chunks[chunk_id] = chunk

        return list(unique_chunks.values())

    def build(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list]:
        """
        Build the final context.

        Returns:
            context:
                Formatted context string.

            selected_chunks:
                Deduplicated and selected chunks.
        """

        if not chunks or self.max_chunks <= 0:
            return "", []

        # -----------------------------------------------------
        # 1. Deduplicate
        # -----------------------------------------------------

        unique_chunks = self._deduplicate(chunks)

        # -----------------------------------------------------
        # 2. Sort by score
        # -----------------------------------------------------

        unique_chunks.sort(
            key=lambda chunk: float(
                getattr(chunk, "score", 0.0)
            ),
            reverse=True,
        )

        # -----------------------------------------------------
        # 3. Limit number of chunks
        # -----------------------------------------------------

        selected_chunks = unique_chunks[
            :self.max_chunks
        ]

        # -----------------------------------------------------
        # 4. Build context
        # -----------------------------------------------------

        context_parts = []

        for index, chunk in enumerate(
            selected_chunks,
            start=1,
        ):
            metadata = getattr(
                chunk,
                "metadata",
                {},
            ) or {}

            document_name = metadata.get(
                "document_name",
                getattr(
                    chunk,
                    "document_name",
                    "",
                ),
            )

            page_numbers = metadata.get(
                "page_numbers",
                getattr(
                    chunk,
                    "page_numbers",
                    [],
                ),
            )

            section = metadata.get(
                "section",
                getattr(
                    chunk,
                    "section",
                    None,
                ),
            )

            content_type = metadata.get(
                "content_type",
                getattr(
                    chunk,
                    "content_type",
                    "text",
                ),
            )

            text = getattr(
                chunk,
                "text",
                "",
            )

            context_parts.append(
                f"Source {index}\n"
                f"Document: {document_name}\n"
                f"Pages: {page_numbers}\n"
                f"Section: {section or 'N/A'}\n"
                f"Content Type: {content_type}\n"
                f"Text:\n{text}"
            )

        context = "\n\n".join(context_parts)

        return context, selected_chunks


def build_context(
    chunks: list[RetrievedChunk],
) -> str:
    """
    Backward-compatible helper used by RAGGenerator.

    RAGGenerator only needs the context string,
    while ContextBuilder.build() also exposes
    the selected chunks.
    """

    builder = ContextBuilder()

    context, _ = builder.build(chunks)

    return context