from pathlib import Path

from src.ingestion.pipeline import process_pdf
from src.ingestion.chunker import create_chunks
from src.embeddings.embedder import EmbeddingModel
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository


PDF_PATH = Path("data/raw/sample.pdf")


def main():
    print("Processing document...")

    # Step 1: Parse and clean the PDF
    elements = process_pdf(PDF_PATH)

    print(f"Elements: {len(elements)}")

    # Step 2: Convert elements into semantic chunks
    chunks = create_chunks(elements)

    print(f"Chunks: {len(chunks)}")

    # Step 3: Load embedding model
    print("\nLoading embedding model...")

    model = EmbeddingModel()

    # Step 4: Generate embeddings
    texts = [chunk.text for chunk in chunks]

    print("\nGenerating embeddings...")

    embeddings = model.encode(texts)

    print(f"Embedding matrix shape: {embeddings.shape}")

    # Step 5: Connect to Qdrant
    client = get_qdrant_client()

    repository = QdrantRepository(
        client=client,
        collection_name="omnirag_documents",
        vector_size=model.dimension(),
    )

    # Step 6: Create collection if it doesn't exist
    repository.create_collection()

    # Step 7: Prepare records
    records = []

    for chunk, vector in zip(chunks, embeddings):
        records.append(
            {
                "point_id": chunk.chunk_id,
                "vector": vector.tolist(),
                "payload": {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "text": chunk.text,
                    "section": chunk.section,
                    "page_numbers": chunk.page_numbers,
                    "element_ids": chunk.element_ids,
                    "element_types": chunk.element_types,
                    "contains_table": chunk.contains_table,
                    "contains_image": chunk.contains_image,
                },
            }
        )

    # Step 8: Upload vectors to Qdrant
    print("\nUploading vectors to Qdrant...")

    repository.upsert_vectors(records)

    print(f"Indexed {len(records)} chunks successfully.")


if __name__ == "__main__":
    main()