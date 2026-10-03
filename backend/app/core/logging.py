"""
Structured logging module for TRUSTTRACE.
Provides structured, audit-ready log entries.
"""
import logging
import sys
from typing import Any, Dict


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures structured console logging for the application."""
    logger = logging.getLogger("trusttrace")
    logger.setLevel(level)

    # Avoid duplicate handlers on reload
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
