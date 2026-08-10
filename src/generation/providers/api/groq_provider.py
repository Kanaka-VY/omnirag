import os

from dotenv import load_dotenv
from groq import Groq

from src.generation.llm import LLM


load_dotenv()


class GroqProvider(LLM):
    """Groq implementation of the OmniRAG LLM interface."""

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv(
            "LLM_MODEL",
            "llama-3.3-70b-versatile",
        )

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content or ""