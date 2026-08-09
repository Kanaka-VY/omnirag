from types import SimpleNamespace

from src.ingestion.multimodal import (
    classify_element,
    represent_table,
    represent_element,
)
def test_represent_table_adds_table_context():
    element = SimpleNamespace(
        element_type="Table",
        text=(
            "Employee | Department | Rating | Salary\n"
            "Ravi | AI | 4.5 | 60000"
        ),
    )

    result = represent_element(element)

    assert result.startswith("Table:")
    assert "Employee" in result
    assert "Ravi" in result
    assert "60000" in result