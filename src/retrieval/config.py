import os
from dataclasses import dataclass


@dataclass
class RetrievalConfig:

    top_k: int = 5

    candidate_k: int = 20

    score_threshold: float | None = 0.0