from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.embeddings.embedder import EmbeddingModel
from src.ingestion.chunker import create_chunks
from src.ingestion.pipeline import process_pdf
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository
from src.evaluation.rag_runner import refresh_retrieval_index


router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    return _embedding_model


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    multimodal: bool = False,
):
    """
    Upload a PDF and ingest it into Qdrant.

    Pipeline:

        PDF
        -> Unstructured parsing
        -> cleaning / normalization
        -> chunking
        -> embeddings
        -> Qdrant
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # ---------------------------------------------------------
    # Save uploaded PDF
    # ---------------------------------------------------------

    document_id = (
        Path(file.filename).stem
        + "_"
        + uuid4().hex[:8]
    )

    safe_filename = (
        f"{document_id}.pdf"
    )

    file_path = UPLOAD_DIR / safe_filename

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        file_path.write_bytes(content)

        # -----------------------------------------------------
        # PDF -> normalized elements
        # -----------------------------------------------------

        elements = process_pdf(
            file_path=file_path,
            multimodal=multimodal,
        )

        if not elements:
            raise HTTPException(
                status_code=422,
                detail="No document elements were extracted.",
            )

        # -----------------------------------------------------
        # Elements -> chunks
        # -----------------------------------------------------

        chunks = create_chunks(elements)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No chunks were created from the document.",
            )

        # -----------------------------------------------------
        # Generate embeddings
        # -----------------------------------------------------

        model = get_embedding_model()

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = model.encode(texts)

        # -----------------------------------------------------
        # Qdrant
        # -----------------------------------------------------

        client = get_qdrant_client()

        repository = QdrantRepository(
            client=client,
            collection_name=QDRANT_COLLECTION,
            vector_size=model.dimension(),
        )

        repository.create_collection()

        records = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            records.append(
                {
                    "point_id": chunk.chunk_id,
                    "vector": embedding.tolist(),
                    "payload": {
                        "text": chunk.text,
                        "document_id": chunk.document_id,
                        "document_name": file.filename,
                        "section": chunk.section,
                        "page_numbers": chunk.page_numbers,
                        "element_ids": chunk.element_ids,
                        "element_types": chunk.element_types,
                        "content_type": chunk.content_type,
                        "table_data": chunk.table_data,
                        "contains_table": chunk.contains_table,
                        "contains_image": chunk.contains_image,
                    },
                }
            )

        repository.upsert_vectors(records)
        refresh_retrieval_index()


        return {
            "status": "success",
            "document_id": document_id,
            "document_name": file.filename,
            "elements": len(elements),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "vector_dimension": model.dimension(),
            "qdrant_collection": QDRANT_COLLECTION,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc
