import asyncio

from src.evaluation.ragas_evaluator import (
    evaluate_faithfulness,
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
    }

    # We will connect the actual evaluator LLM here
    # after verifying the RAGAS metric itself.

    print(record)


if __name__ == "__main__":
    asyncio.run(main())