import re

from rank_bm25 import BM25Okapi

from src.retrieval.models import RetrievedChunk


class BM25Retriever:
    """
    BM25 lexical retriever.

    Uses document chunks from Qdrant as its source of truth.
    The index can be refreshed when new documents are uploaded.
    """

    def __init__(
        self,
        documents: list[dict],
    ):
        self.documents: list[dict] = []
        self.bm25 = None

        self.refresh(documents)

    def refresh(
        self,
        documents: list[dict],
    ) -> None:
        """
        Rebuild the BM25 index from the latest documents.
        """

        self.documents = documents

        tokenized_documents = [
            self._tokenize(
                document.get("text", "")
            )
            for document in documents
        ]

        if not tokenized_documents:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        return re.findall(
            r"\b\w[\w-]*\b",
            text.lower(),
        )
    def refresh(
        self,
        documents: list[dict],
    ) -> None:
        """
        Rebuild the BM25 index from the latest documents.
        """

        self.documents = documents

        tokenized_documents = [
            self._tokenize(
                document.get("text", "")
            )
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:

        if not query.strip():
            return []

        if top_k <= 0:
            return []

        if not self.documents or self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:

            document = self.documents[index]

            metadata = document.get(
                "metadata",
                {},
            )

            results.append(
                RetrievedChunk(
                    chunk_id=str(
                        document["chunk_id"]
                    ),
                    score=float(
                        scores[index]
                    ),
                    text=document.get(
                        "text",
                        "",
                    ),
                    document_id=document.get(
                        "document_id",
                        metadata.get(
                            "document_id",
                            "",
                        ),
                    ),
                    document_name=document.get(
                        "document_name",
                        metadata.get(
                            "document_name",
                            "",
                        ),
                    ),
                    section=document.get(
                        "section",
                        metadata.get(
                            "section",
                        ),
                    ),
                    page_numbers=document.get(
                        "page_numbers",
                        metadata.get(
                            "page_numbers",
                            [],
                        ),
                    ),
                    element_types=document.get(
                        "element_types",
                        metadata.get(
                            "element_types",
                            [],
                        ),
                    ),
                    content_type=document.get(
                        "content_type",
                        metadata.get(
                            "content_type",
                            "text",
                        ),
                    ),
                    table_data=document.get(
                        "table_data",
                        metadata.get(
                            "table_data",
                        ),
                    ),
                    contains_table=document.get(
                        "contains_table",
                        metadata.get(
                            "contains_table",
                            False,
                        ),
                    ),
                    contains_image=document.get(
                        "contains_image",
                        metadata.get(
                            "contains_image",
                            False,
                        ),
                    ),
                    metadata=metadata,
                )
            )

        return results
