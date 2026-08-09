from src.retrieval.lexical import BM25Retriever


def test_bm25_retrieves_exact_term():
    documents = [
        {
            "chunk_id": "1",
            "text": "Employees receive annual leave.",
        },
        {
            "chunk_id": "2",
            "text": "Employee ID EMP-2025-0042 belongs to Ravi.",
        },
        {
            "chunk_id": "3",
            "text": "The company provides health insurance.",
        },
    ]

    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "EMP-2025-0042",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "2"

def test_bm25_returns_ranked_results():
    documents = [
        {
            "chunk_id": "1",
            "text": "Python programming language.",
        },
        {
            "chunk_id": "2",
            "text": "Python machine learning tutorial.",
        },
        {
            "chunk_id": "3",
            "text": "Cooking recipes and ingredients.",
        },
    ]

    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "Python machine learning",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "2"