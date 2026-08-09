from src.ingestion.cleaner import (
    clean_text,
    should_remove_element,
)


def test_clean_text_removes_extra_spaces():
    text = "The    company     reported   growth."

    result = clean_text(text)

    assert result == "The company reported growth."


def test_clean_text_removes_excessive_blank_lines():
    text = "First paragraph.\n\n\n\nSecond paragraph."

    result = clean_text(text)

    assert result == "First paragraph.\n\nSecond paragraph."


def test_empty_text_should_be_removed():
    assert should_remove_element("NarrativeText", "") is True


def test_page_number_should_be_removed():
    assert should_remove_element("PageNumber", "12") is True


def test_normal_text_should_be_kept():
    assert (
        should_remove_element(
            "NarrativeText",
            "Employees receive annual leave.",
        )
        is False
    )