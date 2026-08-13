import json
from pathlib import Path

from src.evaluation.rag_runner import run_rag


# =========================================================
# Configuration
# =========================================================

OUTPUT_PATH = Path(
    "data/evaluation/evaluation_records.json"
)


# =========================================================
# Evaluation questions
# =========================================================

EVALUATION_QUESTIONS = [
    {
        "question": "What is Ram salary?",
        "reference": "Ram's salary is 60000.",
    },
    {
        "question": "Where does Ram work?",
        "reference": (
            "The documents do not specify where Ram works. "
            "They only identify his department as Engineering."
        ),
    },
    {
        "question": "What is this document about?",
        "reference": (
            "The document contains information about "
            "e-commerce sales data analysis using Python, "
            "SQL, Excel, and Power BI, including sales "
            "trends, customer behavior, and business insights."
        ),
    },
]


# =========================================================
# Generate records
# =========================================================

def generate_evaluation_records() -> list[dict]:

    records = []

    for index, item in enumerate(
        EVALUATION_QUESTIONS,
        start=1,
    ):

        question = item["question"]
        reference = item["reference"]

        print()
        print("=" * 60)
        print(f"Evaluating question {index}")
        print(f"Question: {question}")
        print("=" * 60)

        result = run_rag(question)

        record = {
            "question": question,
            "reference": reference,
            "retrieved_contexts": result.get(
                "retrieved_contexts",
                [],
            ),
            "response": result.get(
                "answer",
                "",
            ),
            "retrieved_context_ids": result.get(
                "retrieved_context_ids",
                [],
            ),
            "citations": result.get(
                "citations",
                [],
            ),
        }

        records.append(record)

        print(
            f"Retrieved contexts: "
            f"{len(record['retrieved_contexts'])}"
        )

        print(
            f"Retrieved chunks: "
            f"{len(record['retrieved_context_ids'])}"
        )

        print(
            f"Answer: "
            f"{record['response']}"
        )


    return records


# =========================================================
# Save records
# =========================================================

def save_records(
    records: list[dict],
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("Evaluation dataset generated successfully.")
    print(f"Records: {len(records)}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    records = generate_evaluation_records()

    save_records(records)