from src.embeddings.embedder import EmbeddingModel
from src.retrieval.retriever import Retriever
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository


COLLECTION_NAME = "omnirag_documents"


def main():
    query = input("Enter your query: ")

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    client = get_qdrant_client()

    repository = QdrantRepository(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        repository=repository,
        top_k=5,
    )

    results = retriever.retrieve(query)

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("-" * 70)

        print(f"Score: {result.get('score')}")
        print(f"Chunk ID: {result.get('id')}")

        payload = result.get("payload", {})

        print(f"Document: {payload.get('document_name')}")
        print(f"Section: {payload.get('section')}")
        print(f"Pages: {payload.get('page_numbers')}")
        print(f"Text: {payload.get('text')}")


if __name__ == "__main__":
    main()