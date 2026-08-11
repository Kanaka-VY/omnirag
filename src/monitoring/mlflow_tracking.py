import mlflow
from typing import Any


EXPERIMENT_NAME = "OmniRAG-RAG-Pipeline"


def setup_mlflow() -> None:
    """
    Configure the MLflow experiment used by OmniRAG.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_rag_run(
    query: str,
    model: str | None = None,
    top_k: int | None = None,
):
    """
    Start an MLflow run for one RAG query.
    """

    run = mlflow.start_run()

    mlflow.log_param(
        "query",
        query,
    )

    if model is not None:
        mlflow.log_param(
            "model",
            model,
        )

    if top_k is not None:
        mlflow.log_param(
            "top_k",
            top_k,
        )

    return run


def log_rag_metrics(
    retrieval_latency: float | None = None,
    generation_latency: float | None = None,
    total_latency: float | None = None,
    num_retrieved_chunks: int | None = None,
) -> None:
    """
    Log runtime metrics for a RAG request.
    """

    if retrieval_latency is not None:
        mlflow.log_metric(
            "retrieval_latency_seconds",
            retrieval_latency,
        )

    if generation_latency is not None:
        mlflow.log_metric(
            "generation_latency_seconds",
            generation_latency,
        )

    if total_latency is not None:
        mlflow.log_metric(
            "total_latency_seconds",
            total_latency,
        )

    if num_retrieved_chunks is not None:
        mlflow.log_metric(
            "num_retrieved_chunks",
            num_retrieved_chunks,
        )


def log_rag_metadata(
    retrieved_context_ids: list[str] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> None:
    """
    Store retrieval trace information as MLflow tags.
    """

    if retrieved_context_ids is not None:
        mlflow.set_tag(
            "retrieved_context_ids",
            ",".join(
                str(chunk_id)
                for chunk_id in retrieved_context_ids
            ),
        )

    if citations is not None:
        mlflow.set_tag(
            "citations",
            str(citations),
        )


def end_rag_run() -> None:
    """
    Finish the active MLflow run.
    """
    if mlflow.active_run() is not None:
        mlflow.end_run()