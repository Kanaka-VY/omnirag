from src.embeddings.embedder import EmbeddingModel
from src.generation.generator import RAGGenerator
from src.generation.llm import LLM
from src.retrieval.retriever import Retriever
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.repository import QdrantRepository


COLLECTION_NAME = "omnirag_documents"


class ConsoleLLM(LLM):
    """
    Temporary console LLM implementation.

    This allows us to test the complete RAG pipeline
    without adding a new API provider yet.
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        print("\n--- SYSTEM PROMPT ---")
        print(system_prompt)

        print("\n--- USER PROMPT ---")
        print(user_prompt)

        return (
            "LLM provider is not configured yet. "
            "The retrieved context was successfully "
            "prepared for generation."
        )


def main() -> None:

    # ---------------------------------------------------------
    # 1. Load embedding model
    # ---------------------------------------------------------

    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    # ---------------------------------------------------------
    # 2. Connect to Qdrant
    # ---------------------------------------------------------

    print("Connecting to Qdrant...")

    client = get_qdrant_client()

    print("Qdrant connected.")

    # ---------------------------------------------------------
    # 3. Create repository
    # ---------------------------------------------------------

    repository = QdrantRepository(
        client=client,
        collection_name=COLLECTION_NAME,
        vector_size=embedding_model.dimension(),
    )

    # ---------------------------------------------------------
    # 4. Create retriever
    # ---------------------------------------------------------

    retriever = Retriever(
        embedding_model=embedding_model,
        repository=repository,
    )

    # ---------------------------------------------------------
    # 5. Create LLM
    # ---------------------------------------------------------

    llm = ConsoleLLM()

    generator = RAGGenerator(
        llm=llm,
    )

    # ---------------------------------------------------------
    # 6. Interactive chat
    # ---------------------------------------------------------

    print("\nOmniRAG Chat")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        query = input("\nYou: ").strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            print("Goodbye!")
            break

        if not query:
            continue

        # -----------------------------------------------------
        # Retrieve relevant chunks
        # -----------------------------------------------------

        print("\nSearching Qdrant...")

        chunks = retriever.retrieve(
            query=query,
            top_k=5,
        )

        print(
            f"Retrieved {len(chunks)} chunks."
        )

        # -----------------------------------------------------
        # Generate grounded answer
        # -----------------------------------------------------

        result = generator.generate(
            query=query,
            chunks=chunks,
        )

        print("\nAssistant:")
        print(result.answer)

        # -----------------------------------------------------
        # Sources
        # -----------------------------------------------------

        print("\nSources:")

        if not result.sources:
            print("- No sources found.")
            continue

        for source in result.sources:

            print(
                f"- {source.document_name}, "
                f"pages {source.page_numbers}, "
                f"score {source.score:.4f}"
            )


if __name__ == "__main__":
    main()