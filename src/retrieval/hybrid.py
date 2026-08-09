from dataclasses import dataclass

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever import RetrievedChunk


@dataclass
class HybridResult:
    chunk_id: str
    score: float
    text: str


class HybridRetriever:
    """
    Combines dense semantic retrieval with BM25
    lexical retrieval.

    Dense retrieval:
        Finds semantically similar content.

    BM25:
        Finds exact keyword matches.

    The two rankings are combined using
    Reciprocal Rank Fusion (RRF).
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

        if not query.strip():
            return []

        if top_k <= 0:
            return []

        # -----------------------------------------------------
        # Dense retrieval
        # -----------------------------------------------------

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=top_k,
            document_id=document_id,
        )

        # -----------------------------------------------------
        # BM25 retrieval
        # -----------------------------------------------------

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        # -----------------------------------------------------
        # Reciprocal Rank Fusion
        # -----------------------------------------------------

        scores: dict[str, float] = {}
        texts: dict[str, str] = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = str(result.chunk_id)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            texts[chunk_id] = result.text

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = str(result.chunk_id)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            texts[chunk_id] = result.text

        # -----------------------------------------------------
        # Sort by fused score
        # -----------------------------------------------------

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            HybridResult(
                chunk_id=chunk_id,
                score=score,
                text=texts[chunk_id],
            )
            for chunk_id, score in ranked[:top_k]
        ]