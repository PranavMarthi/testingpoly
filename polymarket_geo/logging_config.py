"""
Structured logging configuration.
JSON logs in production, human-readable in development.
"""

from __future__ import annotations

import logging
import logging.config
import sys

from polymarket_geo.config import get_settings


def setup_logging() -> None:
    """Configure logging based on environment."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.env == "production":
        # JSON structured logging for production (parseable by log aggregators)
        try:
            import json_log_formatter

            formatter = json_log_formatter.JSONFormatter()
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)

            root = logging.getLogger()
            root.setLevel(level)
            root.handlers = [handler]
        except ImportError:
            _setup_basic_logging(level)
    else:
        _setup_basic_logging(level)


def _setup_basic_logging(level: int) -> None:
    """Human-readable logging for development."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)
