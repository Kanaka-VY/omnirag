from src.embeddings.embedder import EmbeddingModel
from src.generation.generator import RAGGenerator
from src.generation.providers.api_provider import APIProvider
from src.retrieval.retriever import Retriever
from src.vector_store.qdrant_store import QdrantStore


COLLECTION_NAME = "omnirag_documents"


def main() -> None:

    embedding_model = EmbeddingModel()

    vector_store = QdrantStore()

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        collection_name=COLLECTION_NAME,
    )

    llm = APIProvider()

    generator = RAGGenerator(
        llm=llm
    )

    while True:

        query = input(
            "\nYou: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        chunks = retriever.retrieve(
            query=query,
            top_k=5,
        )

        result = generator.generate(
            query=query,
            chunks=chunks,
        )

        print("\nAssistant:")
        print(result.answer)

        print("\nSources:")

        for source in result.sources:
            print(
                f"- {source.document_name}, "
                f"pages {source.page_numbers}"
            )


if __name__ == "__main__":
    main()