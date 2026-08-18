import time

import mlflow

from src.config.settings import (
    LLM_MODEL,
    QDRANT_COLLECTION,
)

from src.embeddings.embedder import EmbeddingModel

from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository

from src.retrieval.config import RetrievalConfig
from src.retrieval.retriever import Retriever
from src.retrieval.lexical import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.models import RetrievedChunk
from src.retrieval.search_pipeline import SearchPipeline

from src.generation.generator import RAGGenerator
from src.generation.providers.api.groq_provider import GroqProvider
from src.generation.citations import build_citations

from src.monitoring.mlflow_tracking import (
    setup_mlflow,
    start_rag_run,
    log_rag_metrics,
    log_rag_metadata,
    end_rag_run,
)

from src.monitoring.phoenix import (
    trace_rag_query,
    PhoenixSpan,
    add_span_attribute,
)


# =========================================================
# MLflow configuration
# =========================================================

setup_mlflow()


# =========================================================
# Retrieval configuration
# =========================================================

_retrieval_config = RetrievalConfig()


# =========================================================
# Pipeline construction
# =========================================================

_embedding_model = EmbeddingModel()

_qdrant_client = get_qdrant_client()

_repository = QdrantRepository(
    client=_qdrant_client,
    collection_name=QDRANT_COLLECTION,
    vector_size=_embedding_model.dimension(),
)
_repository = QdrantRepository(
    client=_qdrant_client,
    collection_name=QDRANT_COLLECTION,
    vector_size=_embedding_model.dimension(),
)

_repository.create_collection()

# ---------------------------------------------------------
# Dense semantic retriever
# ---------------------------------------------------------

_dense_retriever = Retriever(
    embedding_model=_embedding_model,
    repository=_repository,
    config=_retrieval_config,
)


# ---------------------------------------------------------
# BM25 lexical retriever
#
# Build the initial BM25 index from the current
# Qdrant contents.
# ---------------------------------------------------------

_documents = _repository.get_all_chunks()

_bm25_retriever = BM25Retriever(
    documents=_documents,
)


# ---------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------

_hybrid_retriever = HybridRetriever(
    dense_retriever=_dense_retriever,
    bm25_retriever=_bm25_retriever,
    rrf_k=60,
)


# ---------------------------------------------------------
# Reranker
# ---------------------------------------------------------

_reranker = CrossEncoderReranker()


# ---------------------------------------------------------
# Search pipeline
# ---------------------------------------------------------

_search_pipeline = SearchPipeline(
    hybrid_retriever=_hybrid_retriever,
    reranker=_reranker,
)


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

_llm = GroqProvider()


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

_generator = RAGGenerator(
    llm=_llm,
)


# =========================================================
# Retrieval index refresh
# =========================================================

def refresh_retrieval_index() -> None:
    """
    Refresh the BM25 index from the latest Qdrant data.

    This is required after new documents are uploaded,
    because the BM25 retriever keeps its own in-memory
    index.
    """

    documents = _repository.get_all_chunks()

    _hybrid_retriever.refresh_bm25(
        documents
    )


# =========================================================
# Public RAG interface
# =========================================================

def run_rag(question: str) -> dict:
    """
    Run the complete OmniRAG RAG pipeline.

    Flow:

        Question
            ↓
        Refresh BM25 index
            ↓
        Hybrid retrieval
            ↓
        Cross-encoder reranking
            ↓
        Context construction
            ↓
        Groq LLM generation
            ↓
        Citations
            ↓
        Final answer

    Retrieval configuration is controlled through
    RetrievalConfig / environment variables.

    MLflow tracks:

        - query
        - model
        - top_k
        - retrieval latency
        - generation latency
        - total latency
        - retrieved chunks
        - citations
        - answer

    Phoenix tracks:

        - complete RAG query
        - BM25 refresh
        - retrieval
        - context construction
        - generation
        - citations
        - answer
        - latency
    """

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not question or not question.strip():
        raise ValueError(
            "Question must not be empty."
        )

    question = question.strip()

    # -----------------------------------------------------
    # Resolve configuration
    # -----------------------------------------------------

    model_name = LLM_MODEL

    top_k = _retrieval_config.top_k

    candidate_k = _retrieval_config.candidate_k

    # -----------------------------------------------------
    # Start timing
    # -----------------------------------------------------

    start_time = time.perf_counter()

    # -----------------------------------------------------
    # Start MLflow run
    # -----------------------------------------------------

    start_rag_run(
        query=question,
        model=model_name,
        top_k=top_k,
    )

    # -----------------------------------------------------
    # Start Phoenix root trace
    # -----------------------------------------------------

    with trace_rag_query(
        question,
        attributes={
            "rag.pipeline": "OmniRAG",
            "rag.model": model_name,
            "rag.top_k": top_k,
            "rag.candidate_k": candidate_k,
        },
    ) as root_span:

        try:

            # =================================================
            # Refresh BM25
            # =================================================

            with PhoenixSpan(
                "OmniRAG BM25 Refresh",
            ) as refresh_span:

                refresh_retrieval_index()

                add_span_attribute(
                    refresh_span,
                    "retrieval.bm25_refreshed",
                    True,
                )

            # =================================================
            # Retrieval + reranking
            # =================================================

            retrieval_start = time.perf_counter()

            with PhoenixSpan(
                "OmniRAG Retrieval",
                attributes={
                    "retrieval.candidate_k": candidate_k,
                    "retrieval.top_k": top_k,
                },
            ) as retrieval_span:

                reranked_results = (
                    _search_pipeline.search(
                        query=question,
                        candidate_k=candidate_k,
                        top_k=top_k,
                    )
                )

                add_span_attribute(
                    retrieval_span,
                    "retrieval.result_count",
                    len(reranked_results),
                )

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            )

            add_span_attribute(
                root_span,
                "retrieval.latency_seconds",
                retrieval_latency,
            )

            # =================================================
            # Convert RerankedResult -> RetrievedChunk
            # =================================================

            retrieved_chunks: list[RetrievedChunk] = []

            with PhoenixSpan(
                "OmniRAG Context Construction",
            ) as context_span:

                for result in reranked_results:

                    metadata = (
                        result.metadata
                        or {}
                    )

                    retrieved_chunks.append(
                        RetrievedChunk(
                            chunk_id=str(
                                result.chunk_id
                            ),
                            score=float(
                                result.score
                            ),
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

                add_span_attribute(
                    context_span,
                    "context.chunk_count",
                    len(retrieved_chunks),
                )

                add_span_attribute(
                    context_span,
                    "context.chunk_ids",
                    ",".join(
                        chunk.chunk_id
                        for chunk in retrieved_chunks
                    ),
                )

            # =================================================
            # Generation
            # =================================================

            generation_start = time.perf_counter()

            with PhoenixSpan(
                "OmniRAG Generation",
                attributes={
                    "llm.model": model_name,
                    "llm.provider": "Groq",
                },
            ) as generation_span:

                generated = _generator.generate(
                    query=question,
                    chunks=retrieved_chunks,
                )

                add_span_attribute(
                    generation_span,
                    "output.value",
                    generated.answer,
                )

                add_span_attribute(
                    generation_span,
                    "output.mime_type",
                    "text/plain",
                )

            generation_latency = (
                time.perf_counter()
                - generation_start
            )

            add_span_attribute(
                root_span,
                "generation.latency_seconds",
                generation_latency,
            )

            # =================================================
            # Citations
            # =================================================

            with PhoenixSpan(
                "OmniRAG Citations",
            ) as citation_span:

                citations = build_citations(
                    retrieved_chunks
                )

                add_span_attribute(
                    citation_span,
                    "citation.count",
                    len(citations),
                )

            # =================================================
            # Total latency
            # =================================================

            total_latency = (
                time.perf_counter()
                - start_time
            )

            # =================================================
            # Phoenix metadata
            # =================================================

            add_span_attribute(
                root_span,
                "rag.status",
                "success",
            )

            add_span_attribute(
                root_span,
                "rag.answer",
                generated.answer,
            )

            add_span_attribute(
                root_span,
                "rag.retrieved_chunk_count",
                len(retrieved_chunks),
            )

            add_span_attribute(
                root_span,
                "rag.citation_count",
                len(citations),
            )

            add_span_attribute(
                root_span,
                "rag.total_latency_seconds",
                total_latency,
            )

            # =================================================
            # MLflow metrics
            # =================================================

            log_rag_metrics(
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=total_latency,
                num_retrieved_chunks=len(
                    retrieved_chunks
                ),
            )

            # =================================================
            # MLflow metadata
            # =================================================

            log_rag_metadata(
                retrieved_context_ids=[
                    chunk.chunk_id
                    for chunk in retrieved_chunks
                ],
                citations=citations,
            )

            mlflow.set_tag(
                "answer",
                generated.answer,
            )

            mlflow.set_tag(
                "num_citations",
                str(len(citations)),
            )

            mlflow.set_tag(
                "retrieval_top_k",
                str(top_k),
            )

            mlflow.set_tag(
                "retrieval_candidate_k",
                str(candidate_k),
            )

            # =================================================
            # Final application response
            # =================================================

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

        except Exception as exc:

            # =================================================
            # Phoenix failure metadata
            # =================================================

            add_span_attribute(
                root_span,
                "rag.status",
                "failed",
            )

            add_span_attribute(
                root_span,
                "error.type",
                type(exc).__name__,
            )

            add_span_attribute(
                root_span,
                "error.message",
                str(exc)[:1000],
            )

            # =================================================
            # MLflow failure metadata
            # =================================================

            mlflow.set_tag(
                "status",
                "failed",
            )

            mlflow.set_tag(
                "error_type",
                type(exc).__name__,
            )

            mlflow.set_tag(
                "error",
                str(exc)[:1000],
            )

            raise

        finally:

            # =================================================
            # End MLflow run
            # =================================================

            if mlflow.active_run() is not None:

                active_run = mlflow.active_run()

                run_data = mlflow.get_run(
                    active_run.info.run_id
                ).data

                if (
                    run_data.tags.get("status")
                    != "failed"
                ):
                    mlflow.set_tag(
                        "status",
                        "success",
                    )

                end_rag_run()