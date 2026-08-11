import asyncio
import json
from pathlib import Path

from src.evaluation.rag_runner import run_rag
from src.evaluation.ragas_evaluator import (
    build_evaluator_llm,
    build_evaluator_embeddings,
    evaluate_records,
    calculate_average_faithfulness,
    calculate_average_context_precision,
    calculate_average_context_recall,
    calculate_average_answer_relevancy,
)


QUESTIONS_PATH = Path(
    "data/evaluation/rag_questions.json"
)

RESULTS_PATH = Path(
    "data/evaluation/results/ragas_evaluation.json"
)


# =========================================================
# Load evaluation questions
# =========================================================

def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# =========================================================
# Save partial evaluation progress
# =========================================================

def save_partial_records(
    records: list[dict],
) -> None:

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "num_questions": len(records),
        "status": "partial",
        "records": records,
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Load previously completed records
# =========================================================

def load_partial_records() -> list[dict]:

    if not RESULTS_PATH.exists():
        return []

    try:
        with RESULTS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return data.get(
            "records",
            [],
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return []


# =========================================================
# Main evaluation
# =========================================================

async def main():

    # ---------------------------------------------------------
    # Load evaluation questions
    # ---------------------------------------------------------

    questions = load_questions()

    print(
        f"Loaded {len(questions)} "
        "evaluation questions."
    )

    # ---------------------------------------------------------
    # Load previously completed records
    # ---------------------------------------------------------

    records = load_partial_records()

    completed_questions = {
        record["question"]
        for record in records
        if "question" in record
    }

    if records:
        print(
            f"Resuming evaluation: "
            f"{len(records)} "
            "question(s) already completed."
        )

    # ---------------------------------------------------------
    # Run OmniRAG for each question
    # ---------------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]
        reference = item["reference"]

        # -----------------------------------------------------
        # Skip questions already completed
        # -----------------------------------------------------

        if question in completed_questions:

            print(
                f"\nSkipping Question "
                f"{index}/{len(questions)} "
                "(already completed):"
            )

            print(question)

            continue

        # -----------------------------------------------------
        # Display question
        # -----------------------------------------------------

        print("\n" + "=" * 50)

        print(
            f"Question {index}/{len(questions)}:"
        )

        print(question)

        print("\nRunning OmniRAG...")

        # -----------------------------------------------------
        # Run RAG pipeline
        # -----------------------------------------------------

        try:

            rag_result = run_rag(
                question
            )

        except Exception as exc:

            print(
                "\nOmniRAG failed for this question:"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "\nSaving completed records "
                "before stopping."
            )

            save_partial_records(
                records
            )

            raise

        # -----------------------------------------------------
        # Extract response
        # -----------------------------------------------------

        response = rag_result["answer"]

        retrieved_contexts = (
            rag_result["retrieved_contexts"]
        )

        # -----------------------------------------------------
        # Display response
        # -----------------------------------------------------

        print("\nResponse:")
        print(response)

        # -----------------------------------------------------
        # Display retrieved contexts
        # -----------------------------------------------------

        print("\nRetrieved contexts:")

        for i, context in enumerate(
            retrieved_contexts,
            start=1,
        ):

            print(
                f"{i}. {context}"
            )

        # -----------------------------------------------------
        # Build evaluation record
        # -----------------------------------------------------

        record = {
            "question": question,
            "reference": reference,
            "response": response,
            "retrieved_contexts": (
                retrieved_contexts
            ),
            "retrieved_context_ids": (
                rag_result.get(
                    "retrieved_context_ids",
                    [],
                )
            ),
            "citations": (
                rag_result.get(
                    "citations",
                    [],
                )
            ),
        }

        # -----------------------------------------------------
        # Add completed record
        # -----------------------------------------------------

        records.append(record)

        completed_questions.add(
            question
        )

        # -----------------------------------------------------
        # SAVE IMMEDIATELY
        #
        # This is important because Groq can return 429.
        # -----------------------------------------------------

        save_partial_records(
            records
        )

        print(
            f"\nSaved progress: "
            f"{len(records)}/{len(questions)} "
            "questions."
        )

    # =========================================================
    # Check whether all questions were completed
    # =========================================================

    if len(records) < len(questions):

        print(
            "\nNot all questions were completed."
        )

        print(
            f"Completed: "
            f"{len(records)}/{len(questions)}"
        )

        print(
            "\nRAGAS evaluation will not run "
            "until all questions have responses."
        )

        return

    # =========================================================
    # Build RAGAS evaluator
    # =========================================================

    print("\n" + "=" * 50)

    print(
        "Building RAGAS evaluator..."
    )

    evaluator_llm = (
        build_evaluator_llm()
    )

    evaluator_embeddings = (
        build_evaluator_embeddings()
    )

    # =========================================================
    # Run RAGAS evaluation
    # =========================================================

    print("\n" + "=" * 50)

    print(
        "Running RAGAS evaluation..."
    )

    print(
        "This may take some time because "
        "real LLM evaluation calls are being made."
    )

    try:

        results = await evaluate_records(
            records=records,
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=(
                evaluator_embeddings
            ),
        )

    except Exception as exc:

        print(
            "\nRAGAS evaluation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "\nThe completed RAG records "
            "are still saved."
        )

        save_partial_records(
            records
        )

        raise

    # =========================================================
    # Calculate averages
    # =========================================================

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

    # =========================================================
    # Display RAGAS results
    # =========================================================

    print("\n" + "=" * 50)

    print(
        "RAGAS RESULTS"
    )

    print("=" * 50)

    print(
        f"Average Faithfulness: "
        f"{average_faithfulness:.4f}"
    )

    print(
        f"Average Context Precision: "
        f"{average_context_precision:.4f}"
    )

    print(
        f"Average Context Recall: "
        f"{average_context_recall:.4f}"
    )

    print(
        f"Average Answer Relevancy: "
        f"{average_answer_relevancy:.4f}"
    )

    # =========================================================
    # Build final evaluation object
    # =========================================================

    evaluation_output = {
        "num_questions": len(questions),
        "status": "completed",
        "results": results,
        "averages": {
            "faithfulness": (
                average_faithfulness
            ),
            "context_precision": (
                average_context_precision
            ),
            "context_recall": (
                average_context_recall
            ),
            "answer_relevancy": (
                average_answer_relevancy
            ),
        },
    }

    # =========================================================
    # Save final results
    # =========================================================

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation_output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nSaved:",
        RESULTS_PATH,
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())