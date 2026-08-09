from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever


class FakeDenseResult:
    def __init__(
        self,
        chunk_id,
        score,
        text,
    ):
        self.chunk_id = chunk_id
        self.score = score
        self.text = text


class FakeDenseRetriever:

    def retrieve(
        self,
        query,
        top_k=5,
        document_id=None,
    ):
        return [
            FakeDenseResult(
                "2",
                0.90,
                "Ravi works in the AI department.",
            ),
            FakeDenseResult(
                "3",
                0.80,
                "Ravi salary is 60000 INR.",
            ),
        ]


def test_hybrid_retriever_combines_results():

    documents = [
        {
            "chunk_id": "1",
            "text": "Annual leave is 20 days.",
        },
        {
            "chunk_id": "2",
            "text": "Ravi works in the AI department.",
        },
        {
            "chunk_id": "3",
            "text": "Ravi salary is 60000 INR.",
        },
    ]

    bm25 = BM25Retriever(documents)

    dense = FakeDenseRetriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    results = retriever.retrieve(
        "Ravi salary",
        top_k=3,
    )

    assert len(results) == 3

    ids = [
        result.chunk_id
        for result in results
    ]

    assert "3" in ids
    assert "2" in ids


def test_hybrid_retriever_empty_query():

    bm25 = BM25Retriever([])

    dense = FakeDenseRetriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    results = retriever.retrieve("")

    assert results == []


def test_hybrid_retriever_respects_top_k():

    documents = [
        {
            "chunk_id": "1",
            "text": "Ravi works in AI.",
        },
        {
            "chunk_id": "2",
            "text": "Ravi works in HR.",
        },
        {
            "chunk_id": "3",
            "text": "Ravi works in Finance.",
        },
    ]

    bm25 = BM25Retriever(documents)

    dense = FakeDenseRetriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    results = retriever.retrieve(
        "Ravi",
        top_k=2,
    )

    assert len(results) == 2