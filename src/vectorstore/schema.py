from dataclasses import dataclass
from typing import Any


@dataclass
class VectorRecord:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]