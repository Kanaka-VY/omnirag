import asyncio

from src.evaluation.rag_runner import run_rag
from src.evaluation.ragas_evaluator import (
    build_evaluator_llm,
    build_evaluator_embeddings,
    evaluate_answer_relevancy,
)


async def main():

    question = "What is Ravi's salary?"

    rag_result = run_rag(question)

    record = {
        "question": question,
        "response": rag_result["answer"],
        "retrieved_contexts": rag_result["retrieved_contexts"],
    }

    evaluator_llm = build_evaluator_llm()
    evaluator_embeddings = build_evaluator_embeddings()

    score = await evaluate_answer_relevancy(
        record,
        evaluator_llm,
        evaluator_embeddings,
    )

    print("=" * 50)
    print("RAGAS ANSWER RELEVANCY")
    print("=" * 50)
    print("Question:", question)
    print("Response:", rag_result["answer"])
    print("Score:", score)


if __name__ == "__main__":
    asyncio.run(main())