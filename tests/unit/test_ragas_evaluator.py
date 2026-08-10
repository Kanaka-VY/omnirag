from src.evaluation.ragas_evaluator import (
    build_evaluation_dataset,
)


def test_build_evaluation_dataset():

    records = [
        {
            "question": "What is Ravi's salary?",
            "retrieved_contexts": [
                "Ravi's salary is 60000."
            ],
            "response": (
                "Ravi's salary is 60000."
            ),
            "reference": (
                "Ravi's salary is 60000."
            ),
        }
    ]

    dataset = build_evaluation_dataset(
        records
    )

    assert len(dataset.samples) == 1
    assert (
        dataset.samples[0].user_input
        == "What is Ravi's salary?"
    )