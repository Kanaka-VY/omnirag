from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder


@dataclass
class RerankedResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class CrossEncoderReranker:
    """
    Reranks retrieved candidates using a cross-encoder.
    """

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates,
        top_k: int = 5,
    ) -> list[RerankedResult]:

        if not query.strip():
            return []

        if not candidates:
            return []

        if top_k <= 0:
            return []

        # -------------------------------------------------
        # Build query-document pairs
        # -------------------------------------------------

        pairs = [
            [query, candidate.text]
            for candidate in candidates
        ]

        # -------------------------------------------------
        # Cross-encoder scoring
        # -------------------------------------------------

        scores = self.model.predict(pairs)

        # -------------------------------------------------
        # Rank candidates
        # -------------------------------------------------

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # -------------------------------------------------
        # Convert to RerankedResult
        # -------------------------------------------------

        results: list[RerankedResult] = []

        for candidate, score in ranked[:top_k]:

            results.append(
                RerankedResult(
                    chunk_id=str(candidate.chunk_id),
                    text=candidate.text,
                    score=float(score),
                    metadata=getattr(
                        candidate,
                        "metadata",
                        {},
                    ) or {},
                )
            )

        return results