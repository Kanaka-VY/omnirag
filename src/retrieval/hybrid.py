from dataclasses import dataclass
from typing import Any

from src.retrieval.lexical import BM25Retriever


@dataclass
class HybridResult:
    """
    Result produced by hybrid retrieval.

    Contains the fused RRF score while preserving
    the original chunk metadata for downstream
    reranking, generation, and citations.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class HybridRetriever:
    """
    Combines dense semantic retrieval with BM25
    lexical retrieval using Reciprocal Rank Fusion (RRF).

    Dense retrieval:
        Finds semantically similar chunks.

    BM25 retrieval:
        Finds keyword/exact-term matches.

    RRF:
        Combines the two rankings into one ranking.
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[HybridResult]:
        """
        Retrieve chunks using dense + BM25 retrieval
        and combine their rankings using RRF.
        """

        # -------------------------------------------------
        # Validate query
        # -------------------------------------------------

        if not query.strip():
            return []

        if top_k <= 0:
            return []

        # -------------------------------------------------
        # 1. Dense semantic retrieval
        # -------------------------------------------------

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=top_k,
            document_id=document_id,
        )

        # -------------------------------------------------
        # 2. BM25 lexical retrieval
        # -------------------------------------------------

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        # -------------------------------------------------
        # Keep original results by chunk ID
        # -------------------------------------------------

        dense_by_id = {
            str(result.chunk_id): result
            for result in dense_results
        }

        bm25_by_id = {
            str(result.chunk_id): result
            for result in bm25_results
        }

        # -------------------------------------------------
        # 3. Reciprocal Rank Fusion
        #
        # RRF score:
        #
        #     1 / (rrf_k + rank)
        #
        # If a chunk appears in both rankings,
        # its scores are added.
        # -------------------------------------------------

        scores: dict[str, float] = {}

        # Dense ranking
        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = str(result.chunk_id)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        # BM25 ranking
        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = str(result.chunk_id)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        # -------------------------------------------------
        # 4. Sort by fused RRF score
        # -------------------------------------------------

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # -------------------------------------------------
        # 5. Build HybridResult objects
        #
        # Prefer the dense result when available because
        # RetrievedChunk contains complete metadata:
        #
        # document_id
        # document_name
        # page_numbers
        # content_type
        # etc.
        # -------------------------------------------------

        results: list[HybridResult] = []

        for chunk_id, fused_score in ranked[:top_k]:

            if chunk_id in dense_by_id:

                original = dense_by_id[chunk_id]

                results.append(
                    HybridResult(
                        chunk_id=str(
                            original.chunk_id
                        ),
                        score=float(
                            fused_score
                        ),
                        text=original.text,
                        metadata=original.metadata,
                    )
                )

            elif chunk_id in bm25_by_id:

                original = bm25_by_id[chunk_id]

                results.append(
                    HybridResult(
                        chunk_id=str(
                            original.chunk_id
                        ),
                        score=float(
                            fused_score
                        ),
                        text=original.text,
                        metadata=getattr(
                            original,
                            "metadata",
                            {},
                        ),
                    )
                )

        return results