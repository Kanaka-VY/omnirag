from src.embeddings.embedder import EmbeddingModel
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository

from src.retrieval.retriever import Retriever
from src.retrieval.lexical import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.models import RetrievedChunk
from src.retrieval.search_pipeline import SearchPipeline

from src.generation.generator import RAGGenerator
from src.generation.providers.api.groq_provider import GroqProvider
from src.generation.citations import build_citations


# ---------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------

_embedding_model = EmbeddingModel()

_qdrant_client = get_qdrant_client()

_repository = QdrantRepository(
    client=_qdrant_client,
    collection_name="omnirag_documents",
    vector_size=_embedding_model.dimension(),
)

_dense_retriever = Retriever(
    embedding_model=_embedding_model,
    repository=_repository,
)

# Build BM25 from the same Qdrant source of truth.
_documents = _repository.get_all_chunks()

_bm25_retriever = BM25Retriever(
    documents=_documents,
)

_hybrid_retriever = HybridRetriever(
    dense_retriever=_dense_retriever,
    bm25_retriever=_bm25_retriever,
)

_reranker = CrossEncoderReranker()

_search_pipeline = SearchPipeline(
    hybrid_retriever=_hybrid_retriever,
    reranker=_reranker,
)

_llm = GroqProvider()

_generator = RAGGenerator(
    llm=_llm,
)


# ---------------------------------------------------------
# Public RAG interface
# ---------------------------------------------------------

def run_rag(question: str) -> dict:
    """
    Run the complete OmniRAG pipeline.

    Flow:

        question
            ↓
        hybrid retrieval
            ↓
        reranking
            ↓
        context construction
            ↓
        LLM generation
            ↓
        answer + contexts + IDs + citations
    """

    if not question or not question.strip():
        raise ValueError(
            "Question must not be empty."
        )

    question = question.strip()

    # -----------------------------------------------------
    # Retrieval + reranking
    # -----------------------------------------------------

    reranked_results = _search_pipeline.search(
        query=question,
        candidate_k=20,
        top_k=5,
    )

    # -----------------------------------------------------
    # Convert RerankedResult → RetrievedChunk
    #
    # This restores the metadata contract expected by
    # the generation and citation layers.
    # -----------------------------------------------------

    retrieved_chunks: list[RetrievedChunk] = []

    for result in reranked_results:

        metadata = result.metadata or {}

        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=str(result.chunk_id),
                score=float(result.score),
                text=result.text,
                document_id=str(
                    metadata.get(
                        "document_id",
                        "",
                    )
                ),
                document_name=str(
                    metadata.get(
                        "document_name",
                        "",
                    )
                ),
                section=metadata.get(
                    "section"
                ),
                page_numbers=metadata.get(
                    "page_numbers",
                    [],
                ),
                element_types=metadata.get(
                    "element_types",
                    [],
                ),
                content_type=metadata.get(
                    "content_type",
                    "text",
                ),
                table_data=metadata.get(
                    "table_data"
                ),
                contains_table=metadata.get(
                    "contains_table",
                    False,
                ),
                contains_image=metadata.get(
                    "contains_image",
                    False,
                ),
                metadata=metadata,
            )
        )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    generated = _generator.generate(
        query=question,
        chunks=retrieved_chunks,
    )

    # -----------------------------------------------------
    # Extract citations
    # -----------------------------------------------------

    citations = build_citations(
        retrieved_chunks
    )

    # -----------------------------------------------------
    # Final evaluation/application contract
    # -----------------------------------------------------

    return {
        "question": question,
        "answer": generated.answer,
        "retrieved_contexts": [
            chunk.text
            for chunk in retrieved_chunks
        ],
        "retrieved_context_ids": [
            chunk.chunk_id
            for chunk in retrieved_chunks
        ],
        "citations": citations,
    }