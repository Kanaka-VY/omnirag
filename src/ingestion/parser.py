from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf


def parse_pdf(file_path: Path):
    """
    Parse a PDF into Unstructured document elements.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {file_path.suffix}")

    return partition_pdf(filename=str(file_path))


def element_to_dict(element: Any) -> dict:
    """
    Convert an Unstructured element into a simple
    JSON-serializable representation.
    """
    metadata = element.metadata

    return {
        "element_id": getattr(element, "id", None),
        "type": type(element).__name__,
        "text": str(element),
        "metadata": {
            "filename": getattr(metadata, "filename", None),
            "filetype": getattr(metadata, "filetype", None),
            "page_number": getattr(metadata, "page_number", None),
            "parent_id": getattr(metadata, "parent_id", None),
        },
    }