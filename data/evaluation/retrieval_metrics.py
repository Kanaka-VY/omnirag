from typing import Sequence


def recall_at_k(
    retrieved_texts: Sequence[str],
    expected_text: str,
    k: int,
) -> float:

    top_k = retrieved_texts[:k]

    expected = expected_text.lower()

    for text in top_k:
        if expected in text.lower():
            return 1.0

    return 0.0