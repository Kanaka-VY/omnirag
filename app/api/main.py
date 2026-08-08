import logging

from src.config.settings import LOG_DIR
from src.monitoring.logging_config import setup_logging


setup_logging(LOG_DIR)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("OmniRAG application started.")
    print("OmniRAG foundation is working.")


if __name__ == "__main__":
    main()