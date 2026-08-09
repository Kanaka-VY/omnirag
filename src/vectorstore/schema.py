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
    chunk_id: str
    score: float
    text: str
    document_id: str
    document_name: str
    section: str | None = None

    page_numbers: list[int] = field(
        default_factory=list
    )

    element_types: list[str] = field(
        default_factory=list
    )

    content_type: str = "text"

    table_data: str | None = None

    contains_table: bool = False

    contains_image: bool = False

    metadata: dict = field(
        default_factory=dict
    )

    