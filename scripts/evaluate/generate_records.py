import json
from pathlib import Path

from src.evaluation.rag_runner import run_rag


# =========================================================
# Configuration
# =========================================================

QUESTIONS = [
    "What is Ravi salary?",
    "Where does Ravi work?",
    "What is this document about?",
]

OUTPUT_PATH = Path(
    "data/evaluation/evaluation_records.json"
)


# =========================================================
# Generate evaluation records
# =========================================================

def generate_records() -> list[dict]:

    records = []

    for question in QUESTIONS:

        print()
        print("=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        result = run_rag(question)

        record = {
            "question": question,

            # We don't automatically know the
            # ground-truth reference answer.
            #
            # This must be manually verified.
            "reference": "",

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

        print()
        print("ANSWER:")
        print(record["response"])

        print()
        print(
            f"Retrieved contexts: "
            f"{len(record['retrieved_contexts'])}"
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
    print("=" * 70)
    print(
        f"Saved {len(records)} evaluation records."
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )
    print("=" * 70)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    records = generate_records()

    save_records(records)