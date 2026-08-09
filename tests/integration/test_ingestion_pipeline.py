from pathlib import Path

from src.ingestion.pipeline import process_pdf


def test_process_sample_pdf():
    pdf_path = Path("data/raw/sample.pdf")

    elements = process_pdf(pdf_path)

    assert elements

    for element in elements:
        assert element.document_id == "sample"
        assert element.document_name == "sample.pdf"
        assert element.element_type
        assert element.text