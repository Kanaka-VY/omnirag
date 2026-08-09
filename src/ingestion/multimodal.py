from src.ingestion.chunker import classify_element


def represent_table(element) -> str:
    """
    Convert a table element into a retrieval-friendly
    textual representation.
    """

    text = (element.text or "").strip()

    if not text:
        return "Table content unavailable."

    return f"Table:\n{text}"


def represent_element(element) -> str:
    """
    Convert an element into a representation suitable
    for embedding and retrieval.
    """

    element_kind = classify_element(
        element.element_type
    )

    if element_kind == "table":
        return represent_table(element)

    return (element.text or "").strip()