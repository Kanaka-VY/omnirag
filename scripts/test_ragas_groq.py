import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


load_dotenv()


async def main() -> None:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    # ---------------------------------------------------------
    # 1. Groq OpenAI-compatible async client
    # ---------------------------------------------------------

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # ---------------------------------------------------------
    # 2. RAGAS evaluator LLM
    # ---------------------------------------------------------

    ragas_llm = llm_factory(
        "llama-3.3-70b-versatile",
        provider="openai",
        client=client,
        adapter="instructor",
    )

    print("RAGAS LLM async:", ragas_llm.is_async)

    # ---------------------------------------------------------
    # 3. Manually constructed evaluation record
    # ---------------------------------------------------------

    record = {
        "question": "What is Ravi's salary?",
        "retrieved_contexts": [
            "Ravi's salary is 60000."
        ],
        "response": "Ravi's salary is 60000.",
    }

    # ---------------------------------------------------------
    # 4. Configure metric
    # ---------------------------------------------------------

    metric = Faithfulness(
        llm=ragas_llm,
    )

    # ---------------------------------------------------------
    # 5. Evaluate asynchronously
    # ---------------------------------------------------------

    result = await metric.ascore(
        user_input=record["question"],
        response=record["response"],
        retrieved_contexts=record["retrieved_contexts"],
    )

    # ---------------------------------------------------------
    # 6. Print result
    # ---------------------------------------------------------

    print("\n========== RAGAS FAITHFULNESS ==========")
    print(f"Question: {record['question']}")
    print(f"Response: {record['response']}")
    print(f"Contexts: {record['retrieved_contexts']}")
    print(f"Result: {result}")
    print(f"Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())