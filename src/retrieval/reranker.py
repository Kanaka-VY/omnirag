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

    The reranker:
        1. Scores query-document pairs.
        2. Sorts candidates by cross-encoder score.
        3. Removes candidates below the configured threshold.
        4. Returns at most top_k relevant chunks.
        5. Falls back to the best candidate if the threshold
           removes everything.
    """

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        score_threshold: float = 0.0,
    ):
        self.model = CrossEncoder(model_name)
        self.score_threshold = score_threshold

    def rerank(
        self,
        query: str,
        candidates,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RerankedResult]:

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        if not query or not query.strip():
            return []

        if not candidates:
            return []

        if top_k <= 0:
            return []

        # -------------------------------------------------
        # Resolve threshold
        # -------------------------------------------------

        threshold = (
            self.score_threshold
            if score_threshold is None
            else score_threshold
        )

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

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        # -------------------------------------------------
        # Rank candidates
        # -------------------------------------------------

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # -------------------------------------------------
        # First pass:
        # Keep candidates above threshold.
        # -------------------------------------------------

        filtered: list[RerankedResult] = []

        for candidate, score in ranked:

            score = float(score)

            if score < threshold:
                continue

            filtered.append(
                RerankedResult(
                    chunk_id=str(
                        candidate.chunk_id
                    ),
                    text=candidate.text,
                    score=score,
                    metadata=getattr(
                        candidate,
                        "metadata",
                        {},
                    ) or {},
                )
            )

            if len(filtered) >= top_k:
                break

        # -------------------------------------------------
        # Fallback
        #
        # If no candidate passes the threshold, keep
        # the single best candidate.
        #
        # This protects broad questions such as:
        # "What is this document about?"
        # -------------------------------------------------

        if not filtered and ranked:

            candidate, score = ranked[0]

            filtered.append(
                RerankedResult(
                    chunk_id=str(
                        candidate.chunk_id
                    ),
                    text=candidate.text,
                    score=float(score),
                    metadata=getattr(
                        candidate,
                        "metadata",
                        {},
                    ) or {},
                )
            )

        return filtered