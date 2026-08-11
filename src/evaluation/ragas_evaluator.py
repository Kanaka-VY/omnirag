import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    AnswerRelevancy,
)

async def evaluate_context_precision(
    record: dict,
    evaluator_llm,
) -> float:

    metric = ContextPrecision(
        llm=evaluator_llm
    )

    result = await metric.ascore(
        user_input=record["question"],
        reference=record["reference"],
        retrieved_contexts=record[
            "retrieved_contexts"
        ],
    )

    return float(result.value)

async def evaluate_answer_relevancy(
    record: dict,
    evaluator_llm,
    evaluator_embeddings,
) -> float:

    metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    result = await metric.ascore(
        user_input=record["question"],
        response=record["response"],
    )

    return float(result.value)

load_dotenv()


def build_evaluation_dataset(
    records: list[dict],
) -> EvaluationDataset:
    samples = []

    for record in records:
        samples.append(
            SingleTurnSample(
                user_input=record["question"],
                retrieved_contexts=record["retrieved_contexts"],
                response=record["response"],
                reference=record["reference"],
            )
        )

    return EvaluationDataset(
        samples=samples
    )


def build_evaluator_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    model = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile",
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    return llm_factory(
        model,
        client=client,
    )


async def evaluate_faithfulness(
    record: dict,
    evaluator_llm,
) -> float:

    metric = Faithfulness(
        llm=evaluator_llm
    )

    result = await metric.ascore(
        user_input=record["question"],
        response=record["response"],
        retrieved_contexts=record["retrieved_contexts"],
    )

    return float(result.value)

async def evaluate_records(
    records: list[dict],
    evaluator_llm,
) -> list[dict]:
    results = []

    for record in records:

        faithfulness = await evaluate_faithfulness(
            record,
            evaluator_llm,
        )

        context_precision = await evaluate_context_precision(
            record,
            evaluator_llm,
        )

        results.append(
            {
                "question": record["question"],
                "response": record["response"],
                "retrieved_contexts": record[
                    "retrieved_contexts"
                ],
                "faithfulness": faithfulness,
                "context_precision": context_precision,
            }
        )

    return results

def calculate_average_faithfulness(
    results: list[dict],
) -> float:
    if not results:
        return 0.0

    total = sum(
        result["faithfulness"]
        for result in results
    )

    return total / len(results)