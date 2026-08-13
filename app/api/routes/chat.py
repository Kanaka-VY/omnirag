from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from src.evaluation.rag_runner import run_rag


router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language question about "
            "the uploaded documents."
        ),
    )


class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieved_contexts: list[str]
    retrieved_context_ids: list[str]
    citations: list[dict]


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """
    Ask a question against the indexed OmniRAG documents.

    Flow:

        Question
          -> Hybrid Retrieval
          -> Cross-Encoder Reranking
          -> Context Construction
          -> Groq Generation
          -> Citations

    Observability is handled inside run_rag()
    using MLflow and Phoenix.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    try:

        result = run_rag(question)

        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            retrieved_contexts=result[
                "retrieved_contexts"
            ],
            retrieved_context_ids=result[
                "retrieved_context_ids"
            ],
            citations=result[
                "citations"
            ],
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {exc}",
        ) from exc