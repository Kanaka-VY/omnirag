from src.generation.generator import RAGGenerator
from src.retrieval.models import RetrievedChunk
from src.generation.prompts import build_prompt


class FakeLLM:

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return "Employees receive 20 days of annual leave."


def test_generator_uses_retrieved_context():

    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        score=0.9,
        text=(
            "Employees receive 20 days "
            "of annual leave."
        ),
        document_id="doc-1",
        document_name="employee_handbook.pdf",
        section="Leave Policy",
        page_numbers=[12],
        element_types=["NarrativeText"],
        metadata={},
    )

    generator = RAGGenerator(
        llm=FakeLLM()
    )

    result = generator.generate(
        query="How many leave days do employees get?",
        chunks=[chunk],
    )

    assert (
        result.answer
        == "Employees receive 20 days of annual leave."
    )

    assert len(result.sources) == 1

def test_generator_handles_no_context():

    generator = RAGGenerator(
        llm=FakeLLM()
    )

    result = generator.generate(
        query="What is the salary?",
        chunks=[],
    )

    assert result.sources == []

    assert "could not find" in result.answer.lower()

def test_grounded_prompt_contains_context():

    prompt = build_prompt(
        question="What is Ravi's salary?",
        context="Ravi's salary is 60000.",
    )

    assert "Ravi's salary is 60000." in prompt
    assert "What is Ravi's salary?" in prompt
    assert "ONLY" in prompt
