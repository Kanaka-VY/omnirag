from pathlib import Path

from src.embeddings.embedder import EmbeddingModel
from src.ingestion.chunker import create_chunks
from src.ingestion.pipeline import process_pdf
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository


def index_pdf(pdf_path: Path) -> None:
    print(f"Indexing: {pdf_path}")

    # ---------------------------------------------------------
    # 1. Parse + clean PDF
    # ---------------------------------------------------------

    elements = process_pdf(
        file_path=pdf_path,
        multimodal=False,
    )

    print(f"Elements: {len(elements)}")

    if not elements:
        raise RuntimeError(
            "No elements were extracted from the PDF."
        )

    # ---------------------------------------------------------
    # 2. Create chunks
    # ---------------------------------------------------------

    chunks = create_chunks(
        elements,
        max_characters=1000,
    )

    print(f"Chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError(
            "No chunks were created from the PDF."
        )

    # ---------------------------------------------------------
    # 3. Load embedding model
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel()

    print(
        f"Embedding dimension: "
        f"{embedding_model.dimension()}"
    )

    # ---------------------------------------------------------
    # 4. Create Qdrant repository
    # ---------------------------------------------------------

    repository = QdrantRepository(
        client=get_qdrant_client(),
        collection_name="omnirag_documents",
        vector_size=embedding_model.dimension(),
    )

    repository.create_collection()

    # ---------------------------------------------------------
    # 5. Embed chunk text
    # ---------------------------------------------------------

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(texts)

    # ---------------------------------------------------------
    # 6. Build Qdrant records
    # ---------------------------------------------------------

    records = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

        vector = (
            embedding.tolist()
            if hasattr(embedding, "tolist")
            else list(embedding)
        )

        payload = {
            "text": chunk.text,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "section": chunk.section,
            "page_numbers": chunk.page_numbers,
            "element_ids": chunk.element_ids,
            "element_types": chunk.element_types,
            "content_type": chunk.content_type,
            "table_data": chunk.table_data,
            "contains_table": chunk.contains_table,
            "contains_image": chunk.contains_image,
        }

        records.append(
            {
                "point_id": chunk.chunk_id,
                "vector": vector,
                "payload": payload,
            }
        )

    # ---------------------------------------------------------
    # 7. Upsert into Qdrant
    # ---------------------------------------------------------

    repository.upsert_vectors(records)

    print(
        f"Successfully indexed "
        f"{len(records)} chunks into Qdrant."
    )


if __name__ == "__main__":

    pdf_path = Path(
        "data/raw/evaluation/employees.pdf"
    )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    index_pdf(pdf_path)