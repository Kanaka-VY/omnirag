from pathlib import Path

from src.ingestion.service import (
    DocumentIngestionService,
)


PDF_PATH = Path(
    "data/raw/sample.pdf"
)


def main():
    print("=" * 70)
    print("OmniRAG Document Ingestion")
    print("=" * 70)

    service = DocumentIngestionService()

    print("\nIngesting:", PDF_PATH)

    result = service.ingest_pdf(
        file_path=PDF_PATH,
        multimodal=False,
    )

    print("\n" + "=" * 70)
    print("INGESTION COMPLETED")
    print("=" * 70)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()