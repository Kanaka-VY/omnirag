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

def evaluate_faithfulness(
    record,
    evaluator_llm,
):
    metric = Faithfulness(
        llm=evaluator_llm
    )

    result = metric.score(
        user_input=record["question"],
        response=record["response"],
        retrieved_contexts=record[
            "retrieved_contexts"
        ],
    )

    return float(result.value)