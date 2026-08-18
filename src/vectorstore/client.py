import os

from qdrant_client import QdrantClient


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client.

    Uses Docker service name inside containers
    and localhost when running locally.
    """
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )