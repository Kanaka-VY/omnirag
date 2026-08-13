import asyncio
import json
from pathlib import Path

from src.monitoring.mlflow_tracking import (
    setup_mlflow,
    start_ragas_evaluation_run,
    log_ragas_metrics,
    end_rag_run,
)

from src.evaluation.ragas_evaluator import (
    build_evaluator_llm,
    build_evaluator_embeddings,
    evaluate_records,
    calculate_average_faithfulness,
    calculate_average_context_precision,
    calculate_average_context_recall,
    calculate_average_answer_relevancy,
)


def load_evaluation_records(
    path: str,
) -> list[dict]:
    """Load RAG evaluation records from JSON."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


async def run_evaluation(
    input_path: str,
) -> dict:
    """
    Run the complete RAGAS evaluation pipeline.

    Metrics:
        - Faithfulness
        - Context Precision
        - Context Recall
        - Answer Relevancy
    """

    records = load_evaluation_records(
        input_path
    )

    if not records:
        raise ValueError(
            f"No evaluation records found in: {input_path}"
        )

    # -----------------------------------------------------
    # Build evaluator models
    # -----------------------------------------------------

    evaluator_llm = build_evaluator_llm()

    evaluator_embeddings = (
        build_evaluator_embeddings()
    )

    # -----------------------------------------------------
    # Evaluate records
    # -----------------------------------------------------

    results = await evaluate_records(
        records,
        evaluator_llm,
        evaluator_embeddings,
    )

    # -----------------------------------------------------
    # Calculate averages
    # -----------------------------------------------------

    average_faithfulness = (
        calculate_average_faithfulness(
            results
        )
    )

    average_context_precision = (
        calculate_average_context_precision(
            results
        )
    )

    average_context_recall = (
        calculate_average_context_recall(
            results
        )
    )

    average_answer_relevancy = (
        calculate_average_answer_relevancy(
            results
        )
    )

    # -----------------------------------------------------
    # Final evaluation result
    # -----------------------------------------------------

    return {
        "num_records": len(results),
        "average_faithfulness": (
            average_faithfulness
        ),
        "average_context_precision": (
            average_context_precision
        ),
        "average_context_recall": (
            average_context_recall
        ),
        "average_answer_relevancy": (
            average_answer_relevancy
        ),
        "results": results,
    }


def save_results(
    evaluation: dict,
    output_path: str,
) -> None:
    """Save evaluation results as JSON."""

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=2,
            ensure_ascii=False,
        )

if __name__ == "__main__":

    input_path = (
        "data/evaluation/"
        "evaluation_records.json"
    )

    output_path = (
        "data/evaluation/results/"
        "ragas_results.json"
    )

    # -----------------------------------------------------
    # Configure MLflow
    # -----------------------------------------------------

    setup_mlflow()

    # -----------------------------------------------------
    # Run RAGAS evaluation
    # -----------------------------------------------------

    evaluation = asyncio.run(
        run_evaluation(
            input_path
        )
    )

    # -----------------------------------------------------
    # Save JSON results
    # -----------------------------------------------------

    save_results(
        evaluation,
        output_path,
    )

    # -----------------------------------------------------
    # Start MLflow evaluation run
    # -----------------------------------------------------

    start_ragas_evaluation_run(
        num_records=evaluation[
            "num_records"
        ],
        evaluator_model=(
            "llama-3.1-8b-instant"
        ),
        embedding_model=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        retrieval_type="hybrid_rrf",
        reranker=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
    )

    # -----------------------------------------------------
    # Log RAGAS metrics
    # -----------------------------------------------------

    log_ragas_metrics(
        faithfulness=evaluation[
            "average_faithfulness"
        ],
        context_precision=evaluation[
            "average_context_precision"
        ],
        context_recall=evaluation[
            "average_context_recall"
        ],
        answer_relevancy=evaluation[
            "average_answer_relevancy"
        ],
    )

    # -----------------------------------------------------
    # Finish MLflow run
    # -----------------------------------------------------

    end_rag_run()

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print(
        f"Evaluated "
        f"{evaluation['num_records']} records."
    )

    print(
        "Average Faithfulness: "
        f"{evaluation['average_faithfulness']:.4f}"
    )

    print(
        "Average Context Precision: "
        f"{evaluation['average_context_precision']:.4f}"
    )

    print(
        "Average Context Recall: "
        f"{evaluation['average_context_recall']:.4f}"
    )

    print(
        "Average Answer Relevancy: "
        f"{evaluation['average_answer_relevancy']:.4f}"
    )

    print(
        "MLflow RAGAS evaluation run logged successfully."
    )
