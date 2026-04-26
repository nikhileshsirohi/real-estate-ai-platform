"""Logging utilities for application observability."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


def _json_formatter(record: logging.LogRecord) -> str:
    """Format log records as compact JSON lines."""
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }

    extra_fields = getattr(record, "extra_fields", None)
    if isinstance(extra_fields, dict):
        payload.update(extra_fields)

    return json.dumps(payload, default=str)


class JsonLogFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        return _json_formatter(record)


def setup_logging() -> None:
    """Configure root logging once for the whole application."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger = logging.getLogger()

    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setFormatter(JsonLogFormatter())
        root_logger.setLevel(log_level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a module."""
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **extra_fields: Any) -> None:
    """Emit a structured log entry with extra fields."""
    logger.log(level, message, extra={"extra_fields": extra_fields})
