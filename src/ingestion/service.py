from pathlib import Path
from typing import Any

from src.ingestion.pipeline import process_pdf
from src.ingestion.chunker import create_chunks
from src.embeddings.embedder import EmbeddingModel
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository


COLLECTION_NAME = "omnirag_documents"


class DocumentIngestionService:
    """
    End-to-end document ingestion service.

    Flow:

        PDF
          ↓
        parsing + cleaning
          ↓
        structure-aware chunking
          ↓
        embeddings
          ↓
        Qdrant
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.embedding_model = (
            embedding_model
            or EmbeddingModel()
        )

        self.qdrant_client = (
            get_qdrant_client()
        )

        self.repository = QdrantRepository(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            vector_size=(
                self.embedding_model.dimension()
            ),
        )

        self.repository.create_collection()

    def ingest_pdf(
        self,
        file_path: Path,
        multimodal: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest a PDF into the OmniRAG vector store.

        Returns summary metadata about the ingestion.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        # -------------------------------------------------
        # 1. PDF → cleaned document elements
        # -------------------------------------------------

        elements = process_pdf(
            file_path=file_path,
            multimodal=multimodal,
        )

        if not elements:
            raise ValueError(
                "No document elements were extracted."
            )

        # -------------------------------------------------
        # 2. Elements → structure-aware chunks
        # -------------------------------------------------

        chunks = create_chunks(elements)

        if not chunks:
            raise ValueError(
                "No document chunks were created."
            )

        # -------------------------------------------------
        # 3. Chunks → embeddings
        # -------------------------------------------------

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Number of embeddings does not match "
                "number of chunks."
            )

        # -------------------------------------------------
        # 4. Build Qdrant records
        # -------------------------------------------------

        records = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "text": chunk.text,
                "section": chunk.section,
                "page_numbers": chunk.page_numbers,
                "element_ids": chunk.element_ids,
                "element_types": chunk.element_types,
                "contains_table": chunk.contains_table,
                "contains_image": chunk.contains_image,
                "content_type": chunk.content_type,
                "table_data": chunk.table_data,
                "image_path": chunk.image_path,
                "visual_description": (
                    chunk.visual_description
                ),
            }

            records.append(
                {
                    "point_id": chunk.chunk_id,
                    "vector": embedding.tolist(),
                    "payload": payload,
                }
            )

        # -------------------------------------------------
        # 5. Store in Qdrant
        # -------------------------------------------------

        self.repository.upsert_vectors(
            records
        )

        # -------------------------------------------------
        # 6. Return ingestion summary
        # -------------------------------------------------

        return {
            "document_id": chunks[0].document_id,
            "document_name": chunks[0].document_name,
            "elements": len(elements),
            "chunks": len(chunks),
            "embedding_dimension": (
                self.embedding_model.dimension()
            ),
            "collection": COLLECTION_NAME,
        }