from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lexical import LexicalResult


class FakeDenseRetriever:
    def retrieve(self, query, top_k=20):
        return [
            LexicalResult(
                chunk_id="A",
                text="Chunk A",
                score=0.9,
                metadata={},
            ),
            LexicalResult(
                chunk_id="B",
                text="Chunk B",
                score=0.8,
                metadata={},
            ),
            LexicalResult(
                chunk_id="C",
                text="Chunk C",
                score=0.7,
                metadata={},
            ),
        ]


class FakeLexicalRetriever:
    def retrieve(self, query, top_k=20):
        return [
            LexicalResult(
                chunk_id="C",
                text="Chunk C",
                score=8.0,
                metadata={},
            ),
            LexicalResult(
                chunk_id="D",
                text="Chunk D",
                score=7.0,
                metadata={},
            ),
            LexicalResult(
                chunk_id="A",
                text="Chunk A",
                score=6.0,
                metadata={},
            ),
        ]


def test_rrf_combines_rankings():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        lexical_retriever=FakeLexicalRetriever(),
    )

    results = retriever.retrieve(
        "test query",
        top_k=4,
    )

    assert len(results) == 4

    chunk_ids = [
        result.chunk_id
        for result in results
    ]

    assert "A" in chunk_ids
    assert "C" in chunk_ids

def test_rrf_boosts_documents_found_by_both():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        lexical_retriever=FakeLexicalRetriever(),
    )

    results = retriever.retrieve(
        "test query",
        top_k=4,
    )

    scores = {
        result.chunk_id: result.score
        for result in results
    }

    assert scores["A"] > scores["B"]
    assert scores["C"] > scores["D"]