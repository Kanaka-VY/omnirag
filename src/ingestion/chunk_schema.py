from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str

    section: Optional[str] = None

    page_numbers: List[int] = field(
        default_factory=list
    )

    element_ids: List[str] = field(
        default_factory=list
    )

    element_types: List[str] = field(
        default_factory=list
    )

    contains_table: bool = False

    contains_image: bool = False

    # ---------------------------------------------------------
    # Multimodal metadata
    # ---------------------------------------------------------

    content_type: str = "text"

    table_data: Optional[str] = None

    content_type: str = "text"
    table_data: str | None = None
    image_path: str | None = None
    visual_description: str | None = None