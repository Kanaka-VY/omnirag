from qdrant_client import QdrantClient


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client connected
    to the local Qdrant Docker container.
    """
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )