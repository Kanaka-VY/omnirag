from src.retrieval.reranker import (
    CrossEncoderReranker,
)


class FakeModel:

    def predict(
        self,
        pairs,
        show_progress_bar=False,
    ):
        return [
            0.2,
            0.9,
            0.5,
        ]


class Candidate:

    def __init__(
        self,
        chunk_id,
        text,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = {}


def test_reranker_sorts_by_score():

    reranker = object.__new__(
        CrossEncoderReranker
    )
    reranker.score_threshold = float("-inf")

    reranker.model = FakeModel()

    candidates = [
        Candidate("A", "chunk A"),
        Candidate("B", "chunk B"),
        Candidate("C", "chunk C"),
    ]

    results = reranker.rerank(
        query="test query",
        candidates=candidates,
        top_k=3,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        "B",
        "C",
        "A",
    ]