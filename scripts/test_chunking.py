from pathlib import Path

from src.ingestion.chunker import create_chunks
from src.ingestion.pipeline import process_pdf


PDF_PATH = Path("data/raw/sample.pdf")


def main() -> None:
    elements = process_pdf(PDF_PATH)

    chunks = create_chunks(
        elements,
        max_characters=1000,
    )

    print(f"Elements: {len(elements)}")
    print(f"Chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print("\n" + "=" * 70)
        print(f"Chunk {index + 1}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Section: {chunk.section}")
        print(f"Pages: {chunk.page_numbers}")
        print(f"Types: {chunk.element_types}")
        print(f"Length: {len(chunk.text)}")
        print(f"Contains table: {chunk.contains_table}")
        print(f"Contains image: {chunk.contains_image}")
        print(f"\n{chunk.text}")


if __name__ == "__main__":
    main()