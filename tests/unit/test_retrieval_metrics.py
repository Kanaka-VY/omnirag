from src.evaluation.retrieval_metrics import (
    recall_at_k,
    recall_at_k_ids,
    reciprocal_rank,
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

def test_recall_at_k_ids():

    retrieved = [
        "A",
        "B",
        "C",
        "D",
    ]

    relevant = ["C"]

    assert (
        recall_at_k_ids(
            retrieved,
            relevant,
            k=3,
        )
        == 1.0
    )

def test_recall_at_k_ids_when_missing():

    retrieved = [
        "A",
        "B",
        "C",
    ]

    relevant = ["D"]

    assert (
        recall_at_k_ids(
            retrieved,
            relevant,
            k=3,
        )
        == 0.0
    )

def test_reciprocal_rank():

    retrieved = [
        "A",
        "B",
        "C",
    ]

    relevant = ["B"]

    assert (
        reciprocal_rank(
            retrieved,
            relevant,
        )
        == 0.5
    )