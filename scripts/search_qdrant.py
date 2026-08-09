from src.embeddings.embedder import EmbeddingModel
from src.vectorstore.client import get_qdrant_client


COLLECTION_NAME = "omnirag_documents"


def main():
    query = input("Enter your query: ")

    print("\nLoading embedding model...")
    model = EmbeddingModel()

    print("Generating query embedding...")
    query_vector = model.encode([query])[0].tolist()

    client = get_qdrant_client()

    print("Searching Qdrant...\n")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
        with_payload=True,
    )

    points = results.points

    if not points:
        print("No results found.")
        return

    print("=" * 70)

    for index, point in enumerate(points, start=1):
        payload = point.payload or {}

        print(f"Result {index}")
        print(f"Score: {point.score}")
        print(f"Chunk ID: {point.id}")
        print(f"Document: {payload.get('document_name')}")
        print(f"Section: {payload.get('section')}")
        print(f"Pages: {payload.get('page_numbers')}")
        print(f"Text: {payload.get('text')}")

        print("=" * 70)


if __name__ == "__main__":
    main()