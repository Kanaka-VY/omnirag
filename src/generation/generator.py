from src.generation.context import build_context
from src.generation.models import GeneratedAnswer
from src.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.generation.llm import LLM
from src.retrieval.models import RetrievedChunk


class RAGGenerator:
    def __init__(
        self,
        llm: LLM,
    ):
        self.llm = llm

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:

        if not chunks:
            return GeneratedAnswer(
                answer=(
                    "I could not find relevant "
                    "information in the provided documents."
                ),
                sources=[],
            )

        context = build_context(chunks)

        user_prompt = build_user_prompt(
            query=query,
            context=context,
        )

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return GeneratedAnswer(
            answer=answer,
            sources=chunks,
        )