from src.embeddings.embedder import EmbeddingModel
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository
from src.retrieval.retriever import Retriever
from src.retrieval.config import RetrievalConfig
from src.evaluation.retrieval_metrics import recall_at_k


COLLECTION_NAME = "omnirag_documents"


EVALUATION_DATASET = [
    {
        "query": "What are the candidate's technical skills?",
        "expected_text": "Python",
    },
    {
        "query": "What internships did the candidate complete?",
        "expected_text": "Generative AI",
    },
    {
        "query": "What AI project did the candidate develop?",
        "expected_text": "YOLOv8",
    },
]


def main():

    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    print("Connecting to Qdrant...")

    client = get_qdrant_client()

    repository = QdrantRepository(
        client=client,
        collection_name=COLLECTION_NAME,
        vector_size=embedding_model.dimension(),
    )

    print("Qdrant connected.")

    config = RetrievalConfig(
        top_k=5,
        score_threshold=None,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        repository=repository,
        config=config,
    )

    print("\nStarting dense retrieval evaluation...\n")

    scores = []

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1,
    ):

        query = item["query"]
        expected_text = item["expected_text"]

        print("=" * 70)
        print(f"Query {index}: {query}")
        print(f"Expected: {expected_text}")

        results = retriever.retrieve(
            query=query,
            top_k=5,
        )

        retrieved_texts = [
            result.text
            for result in results
        ]

        score = recall_at_k(
            retrieved_texts=retrieved_texts,
            expected_text=expected_text,
            k=5,
        )

        scores.append(score)

        print(f"Retrieved chunks: {len(results)}")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\nRank {rank}"
                f"\nScore: {result.score:.4f}"
                f"\nText: {result.text[:300]}"
            )

        print(f"\nRecall@5: {score:.1f}")

    average_recall = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    print("\n" + "=" * 70)
    print("DENSE RETRIEVAL EVALUATION")
    print("=" * 70)

    print(
        f"Queries evaluated: {len(scores)}"
    )

    print(
        f"Average Recall@5: {average_recall:.3f}"
    )


if __name__ == "__main__":
    main()