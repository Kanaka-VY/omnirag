from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantRepository:
    """
    Handles collection creation, vector insertion,
    and vector search in Qdrant.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "omnirag_documents",
        vector_size: int = 384,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    def create_collection(self) -> None:
        """
        Create the Qdrant collection if it does not already exist.
        """

        collections = self.client.get_collections()

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_vectors(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        """
        Insert or update vectors and their metadata.
        """

        points = [
            PointStruct(
                id=record["point_id"],
                vector=record["vector"],
                payload=record["payload"],
            )
            for record in records
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        document_id: str | None = None,
    ):
        """
        Search for the most similar vectors.

        Optionally restrict the search to a specific document.
        """

        query_filter = None

        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
        )

        return response.points