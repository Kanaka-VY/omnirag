from __future__ import annotations

from src.config.settings import GROQ_API_KEY, LLM_MODEL
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from opentelemetry import trace

from src.generation.llm import LLM
from src.monitoring.phoenix import get_tracer

load_dotenv()


class GroqProvider(LLM):
    """Groq implementation of the OmniRAG LLM interface.

    The actual Groq API call is manually instrumented with
    an OpenTelemetry span so Phoenix can observe the LLM call.
    """

    def __init__(self) -> None:

        api_key = GROQ_API_KEY
        model = LLM_MODEL
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response using Groq.

        The actual API request is traced in Phoenix.
        """

        tracer = get_tracer()

        start_time = time.perf_counter()

        with tracer.start_as_current_span(
            "Groq LLM Call"
        ) as span:

            # -------------------------------------------------
            # Input metadata
            # -------------------------------------------------

            span.set_attribute(
                "llm.provider",
                "Groq",
            )

            span.set_attribute(
                "llm.model",
                self.model,
            )

            span.set_attribute(
                "llm.temperature",
                0.0,
            )

            span.set_attribute(
                "llm.input.mime_type",
                "text/plain",
            )

            # Store prompt lengths rather than duplicating
            # potentially sensitive prompt content.
            span.set_attribute(
                "llm.system_prompt_length",
                len(system_prompt),
            )

            span.set_attribute(
                "llm.user_prompt_length",
                len(user_prompt),
            )

            try:

                # -------------------------------------------------
                # ACTUAL GROQ API CALL
                # -------------------------------------------------

                response = (
                    self.client.chat.completions.create(
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
                )

                # -------------------------------------------------
                # Extract answer
                # -------------------------------------------------

                answer = (
                    response.choices[0]
                    .message
                    .content
                    or ""
                )

                # -------------------------------------------------
                # Token usage
                # -------------------------------------------------

                usage = getattr(
                    response,
                    "usage",
                    None,
                )

                if usage is not None:

                    prompt_tokens = getattr(
                        usage,
                        "prompt_tokens",
                        None,
                    )

                    completion_tokens = getattr(
                        usage,
                        "completion_tokens",
                        None,
                    )

                    total_tokens = getattr(
                        usage,
                        "total_tokens",
                        None,
                    )

                    if prompt_tokens is not None:

                        span.set_attribute(
                            "llm.token_count.prompt",
                            int(prompt_tokens),
                        )

                    if completion_tokens is not None:

                        span.set_attribute(
                            "llm.token_count.completion",
                            int(completion_tokens),
                        )

                    if total_tokens is not None:

                        span.set_attribute(
                            "llm.token_count.total",
                            int(total_tokens),
                        )

                # -------------------------------------------------
                # Output metadata
                # -------------------------------------------------

                span.set_attribute(
                    "llm.output.mime_type",
                    "text/plain",
                )

                span.set_attribute(
                    "llm.output_length",
                    len(answer),
                )

                # -------------------------------------------------
                # Latency
                # -------------------------------------------------

                latency = (
                    time.perf_counter()
                    - start_time
                )

                span.set_attribute(
                    "llm.latency_seconds",
                    latency,
                )

                span.set_attribute(
                    "llm.status",
                    "success",
                )

                return answer

            except Exception as exc:

                # -------------------------------------------------
                # Record failure
                # -------------------------------------------------

                span.set_attribute(
                    "llm.status",
                    "failed",
                )

                span.set_attribute(
                    "error.type",
                    type(exc).__name__,
                )

                span.set_attribute(
                    "error.message",
                    str(exc)[:1000],
                )

                try:

                    span.record_exception(
                        exc
                    )

                except Exception:

                    pass

                try:

                    span.set_status(
                        trace.Status(
                            trace.StatusCode.ERROR,
                            str(exc),
                        )
                    )

                except Exception:

                    pass

                raise