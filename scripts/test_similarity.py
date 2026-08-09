import numpy as np

from src.embeddings.embedder import EmbeddingModel


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    numerator = np.dot(vector_a, vector_b)

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def main() -> None:
    model = EmbeddingModel()

    texts = [
        "Employees are entitled to 20 days of annual leave.",
        "Workers receive twenty vacation days.",
        "The company sells enterprise software.",
    ]

    embeddings = model.encode(texts)

    similarity_01 = cosine_similarity(
        embeddings[0],
        embeddings[1],
    )

    similarity_02 = cosine_similarity(
        embeddings[0],
        embeddings[2],
    )

    print(
        "Similarity between first and second:",
        similarity_01,
    )

    print(
        "Similarity between first and third:",
        similarity_02,
    )


if __name__ == "__main__":
    main()