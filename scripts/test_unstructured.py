import json
from pathlib import Path

from src.ingestion.parser import element_to_dict, parse_pdf


PDF_PATH = Path("data/raw/sample.pdf")
OUTPUT_PATH = Path("data/processed/sample.json")


def main() -> None:
    elements = parse_pdf(PDF_PATH)

    parsed_elements = [
        element_to_dict(element)
        for element in elements
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            parsed_elements,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Parsed {len(parsed_elements)} elements.")
    print(f"Saved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()