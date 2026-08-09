import sys
from pathlib import Path

from src.ingestion.pipeline import process_pdf


DEFAULT_PDF = Path("data/raw/sample.pdf")


def main() -> None:
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = DEFAULT_PDF

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    elements = process_pdf(pdf_path)

    print(f"PDF: {pdf_path}")
    print(f"Total elements: {len(elements)}")

    for index, element in enumerate(
        elements,
        start=1,
    ):
        print("\n" + "=" * 80)

        print(f"Element #{index}")
        print(f"Type: {element.element_type}")
        print(f"ID: {element.element_id}")
        print(f"Page: {element.page_number}")
        print(
            f"Text: {(element.text or '')[:500]}"
        )


if __name__ == "__main__":
    main()