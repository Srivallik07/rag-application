"""Application logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_app.config import LoggingConfig


def configure_logging(config: LoggingConfig) -> None:
    """Configure console and optional file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if config.file is not None:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, config.level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )
