from pathlib import Path

import pytest

from src.ingestion.parser import parse_pdf


def test_parse_pdf_missing_file():
    missing_file = Path("data/raw/does_not_exist.pdf")

    with pytest.raises(FileNotFoundError):
        parse_pdf(missing_file)

def test_parse_sample_pdf():
    pdf_path = Path("data/raw/sample.pdf")

    elements = parse_pdf(pdf_path)

    assert elements
    assert len(elements) > 0