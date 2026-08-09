from src.embeddings.embedder import EmbeddingModel


def main() -> None:
    model = EmbeddingModel()

    texts = [
        "Employees are entitled to 20 days of annual leave.",
        "Workers receive twenty vacation days.",
        "The company sells enterprise software.",
    ]

    embeddings = model.encode(texts)

    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", model.dimension())

    for index, embedding in enumerate(embeddings):
        print("\nText:", texts[index])
        print("First 10 values:", embedding[:10])


if __name__ == "__main__":
    main()