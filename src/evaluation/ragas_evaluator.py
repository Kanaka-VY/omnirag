from ragas import EvaluationDataset, SingleTurnSample


def build_evaluation_dataset(
    records: list[dict],
) -> EvaluationDataset:
    samples = []

    for record in records:
        samples.append(
            SingleTurnSample(
                user_input=record["question"],
                retrieved_contexts=record[
                    "retrieved_contexts"
                ],
                response=record["response"],
                reference=record["reference"],
            )
        )

    return EvaluationDataset(
        samples=samples
    )