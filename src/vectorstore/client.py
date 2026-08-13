from qdrant_client import QdrantClient

from src.config.settings import (
    QDRANT_HOST,
    QDRANT_PORT,
)


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client.

    Configuration is loaded from environment variables
    through src.config.settings.
    """

    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )