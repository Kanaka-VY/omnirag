from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentElement:
    """
    Normalized representation of an element extracted from a document.
    """

    element_id: str
    document_id: str
    document_name: str
    element_type: str
    text: str

    page_number: Optional[int] = None
    parent_id: Optional[str] = None
    section: Optional[str] = None
    text_as_html: Optional[str] = None