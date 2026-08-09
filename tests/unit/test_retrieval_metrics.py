from src.evaluation.retrieval_metrics import (
    recall_at_k,
)


def test_recall_at_k_when_relevant_chunk_exists():
    retrieved = [
        "Company overview",
        "Employee table: Ravi | AI | 4.5 | 60000",
        "Leave policy",
    ]

    score = recall_at_k(
        retrieved_texts=retrieved,
        expected_text="60000",
        k=3,
    )

    assert score == 1.0


def test_recall_at_k_when_relevant_chunk_missing():
    retrieved = [
        "Company overview",
        "Leave policy",
        "Insurance policy",
    ]

    score = recall_at_k(
        retrieved_texts=retrieved,
        expected_text="60000",
        k=3,
    )

    assert score == 0.0