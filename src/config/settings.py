import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------
# Data directories
# ---------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

# ---------------------------------------------------------
# Logs
# ---------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama-3.3-70b-versatile",
)

# ---------------------------------------------------------
# Qdrant configuration
# ---------------------------------------------------------

QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "localhost",
)

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_PORT",
        "6333",
    )
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "omnirag_documents",
)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not configured. "
        "Add it to the .env file."
    )