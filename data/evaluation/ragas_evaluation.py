import json
from pathlib import Path

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample


DATASET_PATH = Path(__file__).parent / "ragas_dataset.json"


def load_dataset():
    """Load the RAGAS evaluation dataset from JSON."""
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    samples = []

    for record in records:
        sample = SingleTurnSample(
            user_input=record["question"],
            retrieved_contexts=record["retrieved_contexts"],
            response=record["response"],
            reference=record["reference"],
        )

        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_evaluation():
    """Run RAGAS evaluation on the test dataset."""
    dataset = load_dataset()

    result = evaluate(dataset)

    return result


if __name__ == "__main__":
    result = run_evaluation()
    print(result)