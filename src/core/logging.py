"""
Logging — Structured logging setup for the Digital Twin system.

Provides a JSON formatter for production use and configuration loading
from config/logging.yaml. No external logging service dependency — the
system must be capable of operating without internet access.
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured, machine-readable log output.

    Each log line is a single JSON object with fields:
        timestamp, level, logger, message, and any extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields (anything set via `extra={}` in logging calls)
        standard_attrs = {
            "name", "msg", "args", "created", "relativeCreated", "exc_info",
            "exc_text", "stack_info", "lineno", "funcName", "pathname",
            "filename", "module", "levelno", "levelname", "processName",
            "process", "threadName", "thread", "message", "taskName",
            "msecs",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(
    config_path: str | Path = "config/logging.yaml",
    default_level: int = logging.INFO,
) -> None:
    """Configure logging from a YAML configuration file.

    Falls back to basic configuration if the config file is not found.

    Args:
        config_path: Path to the logging YAML config file.
        default_level: Default log level if config file is missing.
    """
    config_file = Path(config_path)

    if config_file.exists():
        with config_file.open() as f:
            config = yaml.safe_load(f)

        # Ensure logs directory exists for file handlers
        for handler_config in config.get("handlers", {}).values():
            filename = handler_config.get("filename")
            if filename:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Suppress overly verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
