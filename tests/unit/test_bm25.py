from src.retrieval.bm25 import BM25Retriever


def test_bm25_ranks_matching_document_first():
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

    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "Ravi salary",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].chunk_id == "3"


def test_bm25_returns_empty_for_empty_query():
    documents = [
        {
            "chunk_id": "1",
            "text": "Annual leave is 20 days.",
        },
    ]

    retriever = BM25Retriever(documents)

    results = retriever.retrieve("")

    assert results == []


def test_bm25_respects_top_k():
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

    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "Ravi",
        top_k=2,
    )

    assert len(results) == 2


def test_bm25_returns_zero_score_for_unrelated_document():
    documents = [
        {
            "chunk_id": "1",
            "text": "Annual leave is 20 days.",
        },
        {
            "chunk_id": "2",
            "text": "Ravi salary is 60000 INR.",
        },
    ]

    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "Ravi salary",
        top_k=2,
    )

    assert results[0].chunk_id == "2"
    assert results[0].score > results[1].score