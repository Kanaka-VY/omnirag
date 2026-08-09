from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    document_id: str
    document_name: str
    section: str | None
    page_numbers: list[int]
    element_types: list[str]
    metadata: dict[str, Any]