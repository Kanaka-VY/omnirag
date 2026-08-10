from src.embeddings.embedder import EmbeddingModel
from src.retrieval.config import RetrievalConfig
from src.vectorstore.repository import QdrantRepository
# rom src.vectorstore.schema import RetrievedChunk
from src.retrieval.models import RetrievedChunk

class Retriever:
    """
    Performs semantic retrieval over Qdrant.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        repository: QdrantRepository,
        config: RetrievalConfig | None = None,
    ):
        self.embedding_model = embedding_model
        self.repository = repository
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks.

        Args:
            query:
                User's natural-language question.

            top_k:
                Number of chunks to retrieve.
                If None, uses config.top_k.

            document_id:
                Optional document filter.

            score_threshold:
                Optional minimum similarity score.
                If None, uses config.score_threshold.
        """

        # ---------------------------------------------------------
        # Validate query
        # ---------------------------------------------------------

        if not query.strip():
            return []

        # ---------------------------------------------------------
        # Resolve configuration
        # ---------------------------------------------------------

        top_k = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        score_threshold = (
            score_threshold
            if score_threshold is not None
            else self.config.score_threshold
        )

        # ---------------------------------------------------------
        # Convert query into embedding
        # ---------------------------------------------------------

        query_embedding = self.embedding_model.encode([query])[0]
        if hasattr(query_embedding, "tolist"):
            query_vector = query_embedding.tolist()
        else:
            query_vector = list(query_embedding)

        # ---------------------------------------------------------
        # Search Qdrant
        # ---------------------------------------------------------

        results = self.repository.search(
            query_vector=query_vector,
            limit=top_k,
            document_id=document_id,
        )

        # ---------------------------------------------------------
        # Convert Qdrant results into RetrievedChunk objects
        # ---------------------------------------------------------

        retrieved_chunks: list[RetrievedChunk] = []

        for result in results:

            score = float(result.score)

            # -----------------------------------------------------
            # Optional similarity threshold
            # -----------------------------------------------------

            if (
                score_threshold is not None
                and score < score_threshold
            ):
                continue

            payload = result.payload or {}

            retrieved_chunks.append(
    RetrievedChunk(
        chunk_id=str(result.id),
        score=score,
        text=payload.get("text", ""),

        document_id=payload.get(
            "document_id",
            "",
        ),

        document_name=payload.get(
            "document_name",
            "",
        ),

        section=payload.get(
            "section"
        ),

        page_numbers=payload.get(
            "page_numbers",
            [],
        ),

        element_types=payload.get(
            "element_types",
            [],
        ),

        content_type=payload.get(
            "content_type",
            "text",
        ),

        table_data=payload.get(
            "table_data"
        ),

        contains_table=payload.get(
            "contains_table",
            False,
        ),

        contains_image=payload.get(
            "contains_image",
            False,
        ),

        metadata=payload,
    )
)
                

        return retrieved_chunks