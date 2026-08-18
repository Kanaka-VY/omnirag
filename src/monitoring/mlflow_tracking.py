
import os
import mlflow
from typing import Any


EXPERIMENT_NAME = "OmniRAG-RAG-Pipeline"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)

def setup_mlflow() -> None:
    """
    Configure the MLflow experiment used by OmniRAG.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

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

def start_ragas_evaluation_run(
    num_records: int,
    evaluator_model: str | None = None,
    embedding_model: str | None = None,
    retrieval_type: str | None = None,
    reranker: str | None = None,
):
    """
    Start an MLflow run for a RAGAS evaluation.
    """

    run = mlflow.start_run(
        run_name="RAGAS Evaluation"
    )

    mlflow.log_param(
        "evaluation_type",
        "RAGAS",
    )

    mlflow.log_param(
        "num_records",
        num_records,
    )

    if evaluator_model is not None:
        mlflow.log_param(
            "evaluator_model",
            evaluator_model,
        )

    if embedding_model is not None:
        mlflow.log_param(
            "embedding_model",
            embedding_model,
        )

    if retrieval_type is not None:
        mlflow.log_param(
            "retrieval_type",
            retrieval_type,
        )

    if reranker is not None:
        mlflow.log_param(
            "reranker",
            reranker,
        )

    return run


def log_ragas_metrics(
    faithfulness: float,
    context_precision: float,
    context_recall: float,
    answer_relevancy: float,
) -> None:
    """
    Log aggregate RAGAS evaluation metrics.
    """

    mlflow.log_metric(
        "ragas_faithfulness",
        faithfulness,
    )

    mlflow.log_metric(
        "ragas_context_precision",
        context_precision,
    )

    mlflow.log_metric(
        "ragas_context_recall",
        context_recall,
    )

    mlflow.log_metric(
        "ragas_answer_relevancy",
        answer_relevancy,
    )

    # Overall mean across the four RAGAS metrics.
    mean_score = (
        faithfulness
        + context_precision
        + context_recall
        + answer_relevancy
    ) / 4.0

    mlflow.log_metric(
        "ragas_mean_score",
        mean_score,
    )
def log_rag_response(
    answer: str,
) -> None:
    """
    Log the generated RAG response as an MLflow tag.
    """

    mlflow.set_tag(
        "rag_response",
        answer,
    )
