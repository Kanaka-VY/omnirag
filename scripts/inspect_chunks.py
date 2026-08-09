from pathlib import Path

from src.ingestion.pipeline import process_pdf
from src.ingestion.chunker import create_chunks


PDF_PATH = Path("data/raw/multimodal_test.pdf")


def main():
    print("Processing PDF...")

    elements = process_pdf(PDF_PATH)

    print(f"Elements: {len(elements)}")

    chunks = create_chunks(elements)

    print(f"Chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print("\n" + "=" * 80)
        print(f"CHUNK #{index}")
        print("=" * 80)

        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Content type: {chunk.content_type}")
        print(f"Pages: {chunk.page_numbers}")
        print(f"Element types: {chunk.element_types}")

        print("\nText:")
        print(chunk.text)

        if chunk.table_data is not None:
            print("\nTable data:")
            print(chunk.table_data)


if __name__ == "__main__":
    main()