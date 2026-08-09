from src.embeddings.embedder import EmbeddingModel
from src.retrieval.retriever import Retriever
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository


COLLECTION_NAME = "omnirag_documents"


def main():
    print("Connecting to Qdrant...")

    client = get_qdrant_client()

    repository = QdrantRepository(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    print("Qdrant connected.")

    print("\nLoading embedding model...")
    embedding_model = EmbeddingModel()

    retriever = Retriever(
        embedding_model=embedding_model,
        repository=repository,
    )

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    print("\nSearching Qdrant...")

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    print(f"\nRetrieved {len(results)} chunks.")

    if not results:
        print("No relevant chunks found.")
        return

    for index, result in enumerate(results, start=1):
        print("\n" + "=" * 80)
        print(f"RESULT #{index}")
        print("=" * 80)

        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Document ID: {result.document_id}")
        print(f"Document: {result.document_name}")
        print(f"Section: {result.section}")
        print(f"Pages: {result.page_numbers}")
        print(f"Element types: {result.element_types}")

        print("\nText:")
        print(result.text)


if __name__ == "__main__":
    main()
