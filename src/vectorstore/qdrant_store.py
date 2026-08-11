from qdrant_client import models

def search(
    self,
    collection_name: str,
    query_vector,
    limit: int = 5,
    document_id: str | None = None,
):
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
        collection_name=collection_name,
        query=query_vector.tolist(),
        query_filter=query_filter,
        with_payload=True,
        limit=limit,
    )

    return response.points
