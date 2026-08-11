from __future__ import annotations

from typing import Any

from phoenix.otel import register
from opentelemetry import trace


PROJECT_NAME = "OmniRAG"

_tracer_provider = None
_tracer = None


def setup_phoenix():
    """
    Initialize Phoenix OpenTelemetry tracing.

    Returns:
        OpenTelemetry tracer.
    """

    global _tracer_provider
    global _tracer

    if _tracer is not None:
        return _tracer

    _tracer_provider = register(
        project_name=PROJECT_NAME,
    )

    _tracer = trace.get_tracer(
        "omnirag",
    )

    return _tracer


def get_tracer():
    """
    Return the OmniRAG tracer.

    Phoenix is initialized lazily.
    """

    global _tracer

    if _tracer is None:
        return setup_phoenix()

    return _tracer


class PhoenixSpan:
    """
    Context manager used to create an OmniRAG
    Phoenix tracing span.
    """

    def __init__(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ):
        self.name = name
        self.attributes = attributes or {}

        self._context_manager = None
        self.span = None

    def __enter__(self):
        tracer = get_tracer()

        self._context_manager = (
            tracer.start_as_current_span(
                self.name,
            )
        )

        self.span = (
            self._context_manager.__enter__()
        )

        # Add initial attributes
        for key, value in self.attributes.items():
            add_span_attribute(
                self.span,
                key,
                value,
            )

        return self.span

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        if self.span is not None:

            if exc_value is not None:

                try:
                    self.span.record_exception(
                        exc_value
                    )
                except Exception:
                    pass

                try:
                    self.span.set_status(
                        trace.Status(
                            trace.StatusCode.ERROR,
                            str(exc_value),
                        )
                    )
                except Exception:
                    pass

        if self._context_manager is not None:

            self._context_manager.__exit__(
                exc_type,
                exc_value,
                traceback,
            )

        return False


def trace_rag_query(
    question: str,
    *,
    attributes: dict[str, Any] | None = None,
):
    """
    Create a Phoenix span for a complete RAG query.

    Example:

        with trace_rag_query(
            "What is Ravi's salary?"
        ) as span:

            span.set_attribute(
                "rag.question",
                "What is Ravi's salary?",
            )
    """

    if not question or not question.strip():
        raise ValueError(
            "Question must not be empty."
        )

    initial_attributes = {
        "input.value": question,
        "input.mime_type": "text/plain",
    }

    if attributes:
        initial_attributes.update(
            attributes
        )

    return PhoenixSpan(
        name="OmniRAG Query",
        attributes=initial_attributes,
    )


def add_span_attribute(
    span,
    key: str,
    value: Any,
):
    """
    Safely add one attribute to a Phoenix span.
    """

    if span is None:
        return

    try:
        span.set_attribute(
            key,
            value,
        )
    except Exception:
        pass


def add_span_attributes(
    span,
    attributes: dict[str, Any],
):
    """
    Safely add multiple attributes to a Phoenix span.
    """

    if span is None:
        return

    for key, value in attributes.items():

        add_span_attribute(
            span,
            key,
            value,
        )