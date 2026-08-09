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


def recall_at_k_ids(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Measure whether any relevant document/chunk ID
    appears in the top-k retrieved IDs.

    Returns:
        1.0 if a relevant ID is found.
        0.0 otherwise.
    """

    if k <= 0:
        return 0.0

    if not relevant_ids:
        return 0.0

    top_k_ids = retrieved_ids[:k]

    relevant_set = set(relevant_ids)

    for chunk_id in top_k_ids:
        if chunk_id in relevant_set:
            return 1.0

    return 0.0


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    If the first relevant result appears at rank 1:
        1 / 1 = 1.0

    If it appears at rank 2:
        1 / 2 = 0.5

    If it appears at rank 3:
        1 / 3 = 0.333...

    Returns:
        Reciprocal rank of the first relevant result.
        0.0 if no relevant result is found.
    """

    if not retrieved_ids:
        return 0.0

    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0