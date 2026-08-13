from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str

    section: str | None = None

    page_numbers: list[int] = field(
        default_factory=list
    )

    element_ids: list[str] = field(
        default_factory=list
    )

    element_types: list[str] = field(
        default_factory=list
    )

    contains_table: bool = False
    contains_image: bool = False

    # Multimodal metadata
    content_type: str = "text"
    table_data: str | None = None
    image_path: str | None = None
    visual_description: str | None = None