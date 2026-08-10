import asyncio
import json
from pathlib import Path

from src.evaluation.ragas_evaluator import (
    build_evaluator_llm,
    evaluate_records,
)


def load_evaluation_records(
    path: str,
) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_average_faithfulness(
    results: list[dict],
) -> float:
    if not results:
        return 0.0

    return sum(
        result["faithfulness"]
        for result in results
    ) / len(results)


async def run_evaluation(
    input_path: str,
) -> dict:
    records = load_evaluation_records(input_path)

    evaluator_llm = build_evaluator_llm()

    results = await evaluate_records(
        records,
        evaluator_llm,
    )

    average_faithfulness = (
        calculate_average_faithfulness(results)
    )

    return {
        "num_records": len(results),
        "average_faithfulness": average_faithfulness,
        "results": results,
    }


def save_results(
    evaluation: dict,
    output_path: str,
) -> None:
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
    input_path = "data/evaluation/evaluation_records.json"
    output_path = "data/evaluation/results/faithfulness_results.json"

    evaluation = asyncio.run(
        run_evaluation(input_path)
    )

    save_results(
        evaluation,
        output_path,
    )

    print(
        f"Evaluated {evaluation['num_records']} records."
    )

    print(
        "Average faithfulness: "
        f"{evaluation['average_faithfulness']:.4f}"
    )