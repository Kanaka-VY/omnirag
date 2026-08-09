from dataclasses import dataclass


@dataclass
class HybridResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        lexical_retriever,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[HybridResult]:

        dense_results = (
            self.dense_retriever.retrieve(
                query,
                top_k=candidate_k,
            )
        )

        lexical_results = (
            self.lexical_retriever.retrieve(
                query,
                top_k=candidate_k,
            )
        )

        scores = {}
        result_map = {}

        self._add_results(
            dense_results,
            scores,
            result_map,
        )

        self._add_results(
            lexical_results,
            scores,
            result_map,
        )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            HybridResult(
                chunk_id=chunk_id,
                text=result_map[chunk_id].text,
                score=score,
                metadata=result_map[chunk_id].metadata,
            )
            for chunk_id, score in ranked[:top_k]
        ]

    def _add_results(
        self,
        results,
        scores,
        result_map,
    ):
        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = result.chunk_id

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0
                / (self.rrf_k + rank)
            )

            result_map[chunk_id] = result