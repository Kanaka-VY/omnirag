import json
from pathlib import Path


QUESTIONS_PATH = Path(
    "data/evaluation/rag_questions.json"
)


def load_questions() -> list[dict]:

    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():

    questions = load_questions()

    print(
        f"Loaded {len(questions)} "
        "evaluation questions."
    )


if __name__ == "__main__":
    main()