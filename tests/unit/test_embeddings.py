import numpy as np

from src.embeddings.embedder import EmbeddingModel


def test_embedding_shape():
    model = EmbeddingModel()

    texts = [
        "Employees receive annual leave.",
        "Workers receive vacation days.",
    ]

    embeddings = model.encode(texts)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == model.dimension()


def test_embeddings_are_numeric():
    model = EmbeddingModel()

    embeddings = model.encode(
        ["Employees receive annual leave."]
    )

    assert isinstance(embeddings, np.ndarray)
    assert np.issubdtype(
        embeddings.dtype,
        np.number,
    )