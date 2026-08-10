questions = load_questions()

results = []

for item in questions:

    rag_result = run_rag(
        item["question"]
    )

    record = {
        "question": item["question"],
        "retrieved_contexts": (
            rag_result["retrieved_contexts"]
        ),
        "response": rag_result["answer"],
        "reference": item["reference"],
    }

    faithfulness = (
        evaluate_faithfulness(record)
    )

    results.append(
        {
            "question": item["question"],
            "faithfulness": faithfulness,
        }
    )