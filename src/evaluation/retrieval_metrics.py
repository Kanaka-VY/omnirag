def recall_at_k(
    retrieved_texts: list[str],
    expected_text: str,
    k: int,
) -> float:
    """
    Measure whether the expected information appears
    in the top-k retrieved results.

    Returns:
        1.0 if found.
        0.0 otherwise.
    """

    if k <= 0:
        return 0.0

    if not expected_text:
        return 0.0

    top_k_results = retrieved_texts[:k]

    expected = expected_text.strip().lower()

    for text in top_k_results:
        if expected in text.strip().lower():
            return 1.0

    return 0.0