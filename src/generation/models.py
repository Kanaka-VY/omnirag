from dataclasses import dataclass

from src.retrieval.models import RetrievedChunk


@dataclass
class GeneratedAnswer:
    answer: str
    sources: list[RetrievedChunk]