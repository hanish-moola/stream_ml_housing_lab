"""Logging configuration utilities for the Stream-ML Housing Lab project."""

from __future__ import annotations

import logging
import os
from typing import Optional

_LOGGING_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging.

    Args:
        level: Optional string level (e.g. "INFO", "DEBUG"). If not supplied,
            falls back to the HOUSING_LOG_LEVEL environment variable and ultimately
            to ``INFO``.
    """

    resolved_level = (level or os.getenv("HOUSING_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=resolved_level, format=_LOGGING_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with project defaults."""

    return logging.getLogger(name)
