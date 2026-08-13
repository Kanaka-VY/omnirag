from pathlib import Path
from typing import Any

import os

# ---------------------------------------------------------
# Explicit Tesseract configuration
# ---------------------------------------------------------

TESSERACT_DIR = Path(
    r"C:\Program Files\Tesseract-OCR"
)

TESSERACT_EXE = (
    TESSERACT_DIR / "tesseract.exe"
)

if TESSERACT_EXE.exists():
    os.environ["PATH"] = (
        str(TESSERACT_DIR)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

# Configure the exact pytesseract executable used
# by Unstructured.
try:
    import unstructured_pytesseract.pytesseract as pytesseract

    if TESSERACT_EXE.exists():
        pytesseract.tesseract_cmd = str(
            TESSERACT_EXE
        )

except ImportError:
    pass


from unstructured.partition.pdf import partition_pdf


def parse_pdf(
    file_path: Path,
    multimodal: bool = False,
):
    """
    Parse a PDF into Unstructured document elements.

    Args:
        file_path:
            Path to the PDF file.

        multimodal:
            If True, use high-resolution extraction with
            table and image extraction enabled.

        If False, use standard PDF partitioning.
    """

    # ---------------------------------------------------------
    # Validate file
    # ---------------------------------------------------------

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {file_path.suffix}"
        )

    # ---------------------------------------------------------
    # Standard parsing
    # ---------------------------------------------------------

    if not multimodal:
        return partition_pdf(
            filename=str(file_path),
        )

    # ---------------------------------------------------------
    # Multimodal parsing
    # ---------------------------------------------------------

    return partition_pdf(
        filename=str(file_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_types=[
            "Image",
            "Table",
        ],
    )


def element_to_dict(element: Any) -> dict:
    """
    Convert an Unstructured element into a simple
    JSON-serializable representation.
    """

    metadata = getattr(
        element,
        "metadata",
        None,
    )

    return {
        "element_id": getattr(
            element,
            "id",
            None,
        ),
        "type": type(element).__name__,
        "text": str(element),
        "metadata": {
            "filename": getattr(
                metadata,
                "filename",
                None,
            ),
            "filetype": getattr(
                metadata,
                "filetype",
                None,
            ),
            "page_number": getattr(
                metadata,
                "page_number",
                None,
            ),
            "parent_id": getattr(
                metadata,
                "parent_id",
                None,
            ),
        },
    }