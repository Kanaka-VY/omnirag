import logging

from fastapi import FastAPI

from src.config.settings import LOG_DIR
from src.monitoring.logging_config import setup_logging

from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router


setup_logging(LOG_DIR)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="OmniRAG API",
    description="Multimodal Enterprise RAG Platform",
    version="1.0.0",
)


app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "OmniRAG API",
    }


@app.get("/")
def root():
    return {
        "message": "OmniRAG API is running",
    }