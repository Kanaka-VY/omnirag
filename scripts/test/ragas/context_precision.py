import asyncio

from src.evaluation.ragas_evaluator import (
    build_evaluator_llm,
    evaluate_context_precision,
)


async def main():

    record = {
        "question": "What is Ravi's salary?",
        "response": (
            "Ravi's salary is 60000."
        ),
        "retrieved_contexts": [
            (
                "Ravi\n\n"
                "Department: Engineering\n\n"
                "Salary: 60000"
            ),
            (
                "Priya\n\n"
                "Department: HR\n\n"
                "Salary: 55000"
            ),
            "Employee Information",
        ],
        "reference": (
            "Ravi's salary is 60000."
        ),
    }

    evaluator_llm = (
        build_evaluator_llm()
    )

    score = await evaluate_context_precision(
        record,
        evaluator_llm,
    )

    print("=" * 50)
    print("RAGAS CONTEXT PRECISION")
    print("=" * 50)
    print(
        f"Question: {record['question']}"
    )
    print(
        f"Contexts: "
        f"{record['retrieved_contexts']}"
    )
    print(
        f"Score: {score}"
    )


if __name__ == "__main__":
    asyncio.run(main())