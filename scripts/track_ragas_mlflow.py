import json
from pathlib import Path

import mlflow


RESULTS_PATH = Path(
    "data/evaluation/results/ragas_evaluation.json"
)

EXPERIMENT_NAME = "OmniRAG-RAGAS-Evaluation"


def load_results() -> dict:
    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:

    # ---------------------------------------------------------
    # Load completed RAGAS evaluation
    # ---------------------------------------------------------

    data = load_results()

    averages = data["averages"]

    num_questions = data["num_questions"]

    # ---------------------------------------------------------
    # Configure MLflow experiment
    # ---------------------------------------------------------

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---------------------------------------------------------
    # Start MLflow run
    # ---------------------------------------------------------

    with mlflow.start_run(
        run_name="ragas-evaluation"
    ):

        # -----------------------------------------------------
        # Log parameters
        # -----------------------------------------------------

        mlflow.log_param(
            "num_questions",
            num_questions,
        )

        mlflow.log_param(
            "evaluation_status",
            data.get(
                "status",
                "unknown",
            ),
        )

        # -----------------------------------------------------
        # Log RAGAS metrics
        # -----------------------------------------------------

        mlflow.log_metric(
            "faithfulness",
            averages["faithfulness"],
        )

        mlflow.log_metric(
            "context_precision",
            averages["context_precision"],
        )

        mlflow.log_metric(
            "context_recall",
            averages["context_recall"],
        )

        mlflow.log_metric(
            "answer_relevancy",
            averages["answer_relevancy"],
        )

        # -----------------------------------------------------
        # Log complete evaluation JSON
        # -----------------------------------------------------

        mlflow.log_artifact(
            str(RESULTS_PATH),
            artifact_path="evaluation",
        )

        # -----------------------------------------------------
        # Print run information
        # -----------------------------------------------------

        run = mlflow.active_run()

        print("\n" + "=" * 60)
        print("MLFLOW RUN CREATED")
        print("=" * 60)

        print(
            "Experiment:",
            EXPERIMENT_NAME,
        )

        print(
            "Run ID:",
            run.info.run_id,
        )

        print(
            "Faithfulness:",
            averages["faithfulness"],
        )

        print(
            "Context Precision:",
            averages["context_precision"],
        )

        print(
            "Context Recall:",
            averages["context_recall"],
        )

        print(
            "Answer Relevancy:",
            averages["answer_relevancy"],
        )

        print("=" * 60)


if __name__ == "__main__":
    main()