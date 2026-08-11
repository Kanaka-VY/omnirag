import asyncio
import json
from pathlib import Path

from src.evaluation.ragas_evaluator import (
    build_evaluator_llm,
    evaluate_faithfulness,
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_answer_relevancy,
)

from src.evaluation.rag_runner import run_rag


QUESTIONS_FILE = Path(
    "data/evaluation/rag_questions.json"
)

OUTPUT_FILE = Path(
    "data/evaluation/results/ragas_final.json"
)


# ---------------------------------------------------------
# Evaluator embeddings
# ---------------------------------------------------------

from ragas.embeddings import HuggingFaceEmbeddings


def build_evaluator_embeddings():
    return HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

async def main():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    evaluator_llm = build_evaluator_llm()
    evaluator_embeddings = build_evaluator_embeddings()

    results = []

    for item in questions:

        question = item["question"]
        reference = item["reference"]

        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

        rag_result = run_rag(question)

        record = {
            "question": question,
            "reference": reference,
            "response": rag_result["answer"],
            "retrieved_contexts": rag_result[
                "retrieved_contexts"
            ],
        }

        print("Response:", record["response"])

        faithfulness = await evaluate_faithfulness(
            record,
            evaluator_llm,
        )

        context_precision = await evaluate_context_precision(
            record,
            evaluator_llm,
        )

        context_recall = await evaluate_context_recall(
            record,
            evaluator_llm,
        )

        answer_relevancy = await evaluate_answer_relevancy(
            record,
            evaluator_llm,
            evaluator_embeddings,
        )

        result = {
            **record,
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_relevancy": answer_relevancy,
        }

        results.append(result)

        print("Faithfulness:", faithfulness)
        print("Context Precision:", context_precision)
        print("Context Recall:", context_recall)
        print("Answer Relevancy:", answer_relevancy)

    # -----------------------------------------------------
    # Calculate averages
    # -----------------------------------------------------

    if results:

        average_faithfulness = sum(
            r["faithfulness"]
            for r in results
        ) / len(results)

        average_context_precision = sum(
            r["context_precision"]
            for r in results
        ) / len(results)

        average_context_recall = sum(
            r["context_recall"]
            for r in results
        ) / len(results)

        average_answer_relevancy = sum(
            r["answer_relevancy"]
            for r in results
        ) / len(results)

    else:
        average_faithfulness = 0.0
        average_context_precision = 0.0
        average_context_recall = 0.0
        average_answer_relevancy = 0.0

    output = {
        "results": results,
        "averages": {
            "faithfulness": average_faithfulness,
            "context_precision": average_context_precision,
            "context_recall": average_context_recall,
            "answer_relevancy": average_answer_relevancy,
        },
    }

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("FINAL RAGAS EVALUATION")
    print("=" * 60)

    print(
        "Average Faithfulness:",
        average_faithfulness,
    )

    print(
        "Average Context Precision:",
        average_context_precision,
    )

    print(
        "Average Context Recall:",
        average_context_recall,
    )

    print(
        "Average Answer Relevancy:",
        average_answer_relevancy,
    )

    print(
        "\nSaved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    asyncio.run(main())