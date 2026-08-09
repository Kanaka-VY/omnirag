import json
from pathlib import Path

from src.evaluation.retrieval_metrics import recall_at_k
from src.retrieval.retriever import Retriever


QUESTIONS_PATH = Path(
    "data/evaluation/retrieval_questions.json"
)


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    questions = load_questions()

    retriever = Retriever()

    scores = []

    for item in questions:
        question = item["question"]
        expected_text = item["expected_text"]

        results = retriever.retrieve(
            question,
            top_k=5,
        )

        texts = [
            result.text
            for result in results
        ]

        score = recall_at_k(
            retrieved_texts=texts,
            expected_text=expected_text,
            k=5,
        )

        scores.append(score)

        print("\n" + "=" * 80)
        print(f"Question: {question}")
        print(f"Expected: {expected_text}")
        print(f"Recall@5: {score}")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\nRank {rank}"
            )
            print(
                f"Score: {result.score}"
            )
            print(
                f"Content type: "
                f"{result.metadata.get('content_type')}"
            )
            print(
                f"Text: "
                f"{result.text[:300]}"
            )

    average_recall = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    print("\n" + "=" * 80)
    print(
        f"Average Recall@5: "
        f"{average_recall:.3f}"
    )


if __name__ == "__main__":
    main()