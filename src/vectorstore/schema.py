from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    """
    Represents one vector and its metadata before insertion into Qdrant.
    """

    point_id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class RetrievedChunk:
    """
    Represents a chunk returned by semantic retrieval.
    """

    chunk_id: str
    score: float
    text: str

    document_id: str
    document_name: str

    section: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    element_types: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)