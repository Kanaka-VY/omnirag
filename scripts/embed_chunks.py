from pathlib import Path

from src.ingestion.pipeline import process_pdf
from src.ingestion.chunker import create_chunks
from src.embeddings.embedder import EmbeddingModel


PDF_PATH = Path("data/raw/sample.pdf")


def main():
    print("Processing document...")

    # 1. PDF -> cleaned/normalized document elements
    elements = process_pdf(PDF_PATH)

    print(f"Elements: {len(elements)}")

    # 2. Document elements -> chunks
    chunks = create_chunks(elements)

    print(f"Chunks: {len(chunks)}")

    if not chunks:
        print("No chunks were created.")
        return

    # 3. Load embedding model
    print("\nLoading embedding model...")

    model = EmbeddingModel()

    # 4. Extract chunk text
    texts = [chunk.text for chunk in chunks]

    # 5. Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = model.encode(texts)

    print(f"\nEmbedding matrix shape: {embeddings.shape}")

    # 6. Display results
    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings),
        start=1,
    ):
        print("\n" + "=" * 70)

        print(f"Chunk: {index}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Section: {chunk.section}")
        print(f"Pages: {chunk.page_numbers}")
        print(f"Vector dimension: {len(embedding)}")

        print(f"Contains table: {chunk.contains_table}")
        print(f"Contains image: {chunk.contains_image}")

        print(f"Text: {chunk.text[:300]}")


if __name__ == "__main__":
    main()