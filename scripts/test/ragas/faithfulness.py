import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness


load_dotenv()


async def main() -> None:

    # ---------------------------------------------------------
    # 1. Groq API key
    # ---------------------------------------------------------

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    # ---------------------------------------------------------
    # 2. Async Groq client
    # ---------------------------------------------------------

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # ---------------------------------------------------------
    # 3. RAGAS evaluator LLM
    # ---------------------------------------------------------

    ragas_llm = llm_factory(
        "llama-3.3-70b-versatile",
        provider="openai",
        client=client,
        adapter="instructor",
    )

    # ---------------------------------------------------------
    # 4. RAGAS evaluator embeddings
    # ---------------------------------------------------------

    ragas_embeddings = embedding_factory(
        "huggingface",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    print("RAGAS LLM async:", ragas_llm.is_async)
    print(
        "RAGAS embeddings:",
        type(ragas_embeddings).__name__,
    )

    # ---------------------------------------------------------
    # 5. Evaluation record
    # ---------------------------------------------------------

    record = {
        "question": "What is Ravi's salary?",
        "retrieved_contexts": [
            "Ravi's salary is 60000."
        ],
        "response": "Ravi's salary is 60000.",
    }

    # ---------------------------------------------------------
    # 6. Configure Faithfulness
    # ---------------------------------------------------------

    metric = Faithfulness(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # ---------------------------------------------------------
    # 7. Evaluate
    # ---------------------------------------------------------

    result = await metric.ascore(
        user_input=record["question"],
        response=record["response"],
        retrieved_contexts=record["retrieved_contexts"],
    )

    score = float(result.value)

    # ---------------------------------------------------------
    # 8. Build evaluation output
    # ---------------------------------------------------------

    output = {
        "metric": "faithfulness",
        "results": [
            {
                "question": record["question"],
                "response": record["response"],
                "retrieved_contexts": record[
                    "retrieved_contexts"
                ],
                "score": score,
            }
        ],
    }

    # ---------------------------------------------------------
    # 9. Save result
    # ---------------------------------------------------------

    results_dir = Path(
        "data/evaluation/results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        results_dir
        / "ragas_faithfulness.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # 10. Print result
    # ---------------------------------------------------------

    print(
        "\n========== RAGAS FAITHFULNESS =========="
    )
    print(
        f"Question: {record['question']}"
    )
    print(
        f"Response: {record['response']}"
    )
    print(
        f"Contexts: {record['retrieved_contexts']}"
    )
    print(
        f"Score: {score}"
    )
    print(
        f"Saved evaluation result to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())