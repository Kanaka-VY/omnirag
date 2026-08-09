from pathlib import Path

from src.ingestion.pipeline import process_pdf


PDF_PATH = Path("data/raw/sample.pdf")


def main() -> None:
    elements = process_pdf(PDF_PATH)

    print(f"Total elements: {len(elements)}")

    for index, element in enumerate(elements):
        print("\n" + "=" * 80)

        print(f"Element #{index + 1}")
        print(f"Type: {element.element_type}")
        print(f"ID: {element.element_id}")
        print(f"Page: {element.page_number}")
        print(f"Text: {element.text[:500]}")


if __name__ == "__main__":
    main()

{
    "text": "...",

    "document_id": "...",
    "document_name": "...",

    "section": "...",
    "page_numbers": [...],

    "element_ids": [...],
    "element_types": [...],

    "contains_table": True,
    "contains_image": False,

    "content_type": "table",

    "table_data": "...",
}