import sys
from pathlib import Path

from src.ingestion.pipeline import process_pdf


def main():

    # ---------------------------------------------------------
    # Get PDF path
    # ---------------------------------------------------------

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = Path(
            "data/raw/sample.pdf"
        )

    print(f"Processing: {pdf_path}")

    # ---------------------------------------------------------
    # Normal parsing by default
    # ---------------------------------------------------------

    elements = process_pdf(
        pdf_path,
        multimodal=False,
    )

    print(
        f"\nTotal elements: {len(elements)}"
    )

    # ---------------------------------------------------------
    # Inspect elements
    # ---------------------------------------------------------

    for index, element in enumerate(
        elements,
        start=1,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Element #{index}"
        )

        print(
            f"Type: "
            f"{element.element_type}"
        )

        print(
            f"ID: "
            f"{element.element_id}"
        )

        print(
            f"Page: "
            f"{element.page_number}"
        )

        text = (
            element.text or ""
        ).strip()

        print(
            f"Text: {text[:500]}"
        )

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        metadata = getattr(
            element,
            "metadata",
            None,
        )

        print(
            f"Metadata: {metadata}"
        )


if __name__ == "__main__":
    main()