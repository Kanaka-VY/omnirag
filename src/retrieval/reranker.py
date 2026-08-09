from dataclasses import dataclass

from sentence_transformers import CrossEncoder


@dataclass
class RerankedResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    ):
        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        candidates,
        top_k: int = 5,
    ) -> list[RerankedResult]:

        pairs = [
            [query, candidate.text]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            RerankedResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=float(score),
                metadata=candidate.metadata,
            )
            for candidate, score in ranked[:top_k]
        ]