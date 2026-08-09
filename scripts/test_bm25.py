from src.retrieval.bm25 import BM25Retriever


documents = [
    {
        "chunk_id": "1",
        "text": "Annual leave is 20 days.",
    },
    {
        "chunk_id": "2",
        "text": "Ravi works in the AI department.",
    },
    {
        "chunk_id": "3",
        "text": "Ravi salary is 60000 INR.",
    },
]


def main():
    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "Ravi salary",
        top_k=3,
    )

    print("\nBM25 Results")
    print("=" * 60)

    for result in results:
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Score: {result.score:.4f}")
        print(f"Text: {result.text}")
        print("-" * 60)


if __name__ == "__main__":
    main()