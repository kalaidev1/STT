"""
app/core/logging.py
Enterprise logging configuration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from typing import Any

import structlog

from app.core.config import settings


# -----------------------------------------------------------------------------
# Create logs directory
# -----------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Add application information to every log
# -----------------------------------------------------------------------------

def _add_app_context(
    logger: Any,
    method_name: str,
    event_dict: dict,
) -> dict:
    """Automatically add application metadata to every log."""

    event_dict["service"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    event_dict["env"] = settings.APP_ENV

    return event_dict


# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure Structlog + Python logging."""

    log_level = getattr(
        logging,
        settings.LOG_LEVEL.upper(),
        logging.INFO,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Console renderer
    if settings.is_production():
        console_renderer = structlog.processors.JSONRenderer()
    else:
        console_renderer = structlog.dev.ConsoleRenderer(colors=True)

    # File renderer (always JSON)
    file_renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # -------------------------------------------------------------------------
    # Console formatter
    # -------------------------------------------------------------------------

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=console_renderer,
        foreign_pre_chain=shared_processors,
    )

    # -------------------------------------------------------------------------
    # File formatter
    # -------------------------------------------------------------------------

    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=file_renderer,
        foreign_pre_chain=shared_processors,
    )

    # -------------------------------------------------------------------------
    # Console Handler
    # -------------------------------------------------------------------------

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # -------------------------------------------------------------------------
    # File Handler
    # Creates:
    #
    # logs/
    #   app.log
    #   app.log.2026-06-25
    #   app.log.2026-06-24
    #
    # -------------------------------------------------------------------------

    file_handler = TimedRotatingFileHandler(
        filename=LOG_DIR / "app.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )

    file_handler.setFormatter(file_formatter)

    # -------------------------------------------------------------------------
    # Root Logger
    # -------------------------------------------------------------------------

    root_logger = logging.getLogger()

    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    root_logger.setLevel(log_level)

    # -------------------------------------------------------------------------
    # Reduce noisy third-party logs
    # -------------------------------------------------------------------------

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Logger Factory
# -----------------------------------------------------------------------------

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return configured logger."""

    return structlog.get_logger(name)