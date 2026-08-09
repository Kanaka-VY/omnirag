from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float | None = None