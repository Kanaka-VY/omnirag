import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)

load_dotenv()


# =========================================================
# Evaluator LLM
# =========================================================

def build_evaluator_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    # Use a separate evaluator model so that
    # RAGAS evaluation does not consume the same
    # generation-model quota.
    model = os.getenv(
        "RAGAS_LLM_MODEL",
        "llama-3.1-8b-instant",
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    return llm_factory(
        model,
        client=client,
    )

def build_evaluator_embeddings():
    return embedding_factory(
        provider="huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

# =========================================================
# Evaluation Dataset
# =========================================================

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


# =========================================================
# Faithfulness
# =========================================================

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
        retrieved_contexts=record[
            "retrieved_contexts"
        ],
    )

    return float(result.value)


# =========================================================
# Context Precision
# =========================================================

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


# =========================================================
# Context Recall
# =========================================================

async def evaluate_context_recall(
    record: dict,
    evaluator_llm,
) -> float:

    metric = ContextRecall(
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


# =========================================================
# Answer Relevancy
# =========================================================
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



# =========================================================
# Evaluate All Records
# =========================================================

async def evaluate_records(
    records: list[dict],
    evaluator_llm,
    evaluator_embeddings,
) -> list[dict]:

    results = []

    for record in records:

        faithfulness = await evaluate_faithfulness(
            record,
            evaluator_llm,
        )

        context_precision = (
            await evaluate_context_precision(
                record,
                evaluator_llm,
            )
        )

        context_recall = (
            await evaluate_context_recall(
                record,
                evaluator_llm,
            )
        )

        answer_relevancy = (
            await evaluate_answer_relevancy(
                record,
                evaluator_llm,
                evaluator_embeddings,
            )
        )

        results.append(
            {
                "question": record["question"],
                "reference": record["reference"],
                "response": record["response"],
                "retrieved_contexts": record[
                    "retrieved_contexts"
                ],
                "retrieved_context_ids": record.get(
                    "retrieved_context_ids",
                    [],
                ),
                "citations": record.get(
                    "citations",
                    [],
                ),
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_relevancy": answer_relevancy,
            }
        )

    return results


# =========================================================
# Average Faithfulness
# =========================================================

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


# =========================================================
# Average Context Precision
# =========================================================

def calculate_average_context_precision(
    results: list[dict],
) -> float:

    if not results:
        return 0.0

    total = sum(
        result["context_precision"]
        for result in results
    )

    return total / len(results)


# =========================================================
# Average Context Recall
# =========================================================

def calculate_average_context_recall(
    results: list[dict],
) -> float:

    if not results:
        return 0.0

    total = sum(
        result["context_recall"]
        for result in results
    )

    return total / len(results)


def calculate_average_answer_relevancy(
    results: list[dict],
) -> float:

    if not results:
        return 0.0

    total = sum(
        result["answer_relevancy"]
        for result in results
    )

    return total / len(results)
