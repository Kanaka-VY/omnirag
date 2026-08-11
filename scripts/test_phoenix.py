import sys
import time
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.phoenix import trace_rag_query


print("Starting Phoenix test...")

with trace_rag_query(
    "Phoenix test query"
) as span:

    span.set_attribute(
        "test.value",
        "hello",
    )

    time.sleep(1)

print("Trace created successfully.")