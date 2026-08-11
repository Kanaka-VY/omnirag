import json
from pathlib import Path

from src.embeddings.embedder import EmbeddingModel
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository

from src.retrieval.retriever import Retriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.search_pipeline import SearchPipeline

from src.evaluation.retrieval_metrics import (
    recall_at_k_ids,
    reciprocal_rank,
)


QUESTIONS_FILE = Path(
    "data/evaluation/retrieval_questions.json"
)

OUTPUT_FILE = Path(
    "data/evaluation/results/retrieval_evaluation.json"
)


def build_search_pipeline():

    # ---------------------------------------------------------
    # Embedding model
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel()

    # ---------------------------------------------------------
    # Qdrant repository
    # ---------------------------------------------------------

    repository = QdrantRepository(
        get_qdrant_client(),
        "omnirag_documents",
        embedding_model.dimension(),
    )

    # ---------------------------------------------------------
    # Dense retriever
    # ---------------------------------------------------------

    dense_retriever = Retriever(
        embedding_model=embedding_model,
        repository=repository,
    )

    # ---------------------------------------------------------
    # BM25 retriever
    # ---------------------------------------------------------

    all_chunks = repository.get_all_chunks()

    bm25_retriever = BM25Retriever(
        all_chunks
    )

    # ---------------------------------------------------------
    # Hybrid retriever
    # ---------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    # ---------------------------------------------------------
    # Cross encoder reranker
    # ---------------------------------------------------------

    reranker = CrossEncoderReranker()

    # ---------------------------------------------------------
    # Search pipeline
    # ---------------------------------------------------------

    return SearchPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )


def main():

    # ---------------------------------------------------------
    # Load evaluation questions
    # ---------------------------------------------------------

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    # ---------------------------------------------------------
    # Build retrieval pipeline
    # ---------------------------------------------------------

    search_pipeline = build_search_pipeline()

    results = []

    # ---------------------------------------------------------
    # Evaluate every question
    # ---------------------------------------------------------

    for item in questions:

        question = item["question"]

        relevant_ids = item[
            "relevant_chunk_ids"
        ]

        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

        retrieved = search_pipeline.search(
            query=question,
            candidate_k=20,
            top_k=5,
        )

        retrieved_ids = [
            str(result.chunk_id)
            for result in retrieved
        ]

        retrieved_texts = [
            result.text
            for result in retrieved
        ]

        print("\nRetrieved results:")

        for rank, result in enumerate(
            retrieved,
            start=1,
        ):
            print(
                f"\n{rank}. "
                f"{result.chunk_id}"
            )
            print(
                f"   Score: {result.score}"
            )
            print(
                f"   Text: {result.text[:300]}"
            )

        # -----------------------------------------------------
        # Retrieval metrics
        # -----------------------------------------------------

        recall_1 = recall_at_k_ids(
            retrieved_ids,
            relevant_ids,
            k=1,
        )

        recall_3 = recall_at_k_ids(
            retrieved_ids,
            relevant_ids,
            k=3,
        )

        recall_5 = recall_at_k_ids(
            retrieved_ids,
            relevant_ids,
            k=5,
        )

        mrr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        result = {
            "question": question,
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_contexts": retrieved_texts,
            "recall_at_1": recall_1,
            "recall_at_3": recall_3,
            "recall_at_5": recall_5,
            "mrr": mrr,
        }

        results.append(result)

        print("\nMetrics:")
        print("Recall@1:", recall_1)
        print("Recall@3:", recall_3)
        print("Recall@5:", recall_5)
        print("MRR:", mrr)

    # ---------------------------------------------------------
    # Calculate averages
    # ---------------------------------------------------------

    if results:

        average_recall_1 = sum(
            r["recall_at_1"]
            for r in results
        ) / len(results)

        average_recall_3 = sum(
            r["recall_at_3"]
            for r in results
        ) / len(results)

        average_recall_5 = sum(
            r["recall_at_5"]
            for r in results
        ) / len(results)

        average_mrr = sum(
            r["mrr"]
            for r in results
        ) / len(results)

    else:

        average_recall_1 = 0.0
        average_recall_3 = 0.0
        average_recall_5 = 0.0
        average_mrr = 0.0

    # ---------------------------------------------------------
    # Build final output
    # ---------------------------------------------------------

    output = {
        "results": results,
        "averages": {
            "recall_at_1": average_recall_1,
            "recall_at_3": average_recall_3,
            "recall_at_5": average_recall_5,
            "mrr": average_mrr,
        },
    }

    # ---------------------------------------------------------
    # Save evaluation
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 60)

    print(
        "Average Recall@1:",
        average_recall_1,
    )

    print(
        "Average Recall@3:",
        average_recall_3,
    )

    print(
        "Average Recall@5:",
        average_recall_5,
    )

    print(
        "Average MRR:",
        average_mrr,
    )

    print(
        "\nSaved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()