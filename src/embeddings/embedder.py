from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around a sentence-transformer embedding model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Convert text into embedding vectors.

        Returns:
            NumPy array containing the embedding vectors.
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.
        """
        return self.model.get_sentence_embedding_dimension()