"""Application logging - standard library only.

One logger tree (`jarvis`), a console handler for diagnostics and an
optional file handler. Setup is idempotent; a logging failure (read-only
filesystem, missing directory) is reported and never crashes startup.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from jarvis.core.config import LoggingConfig

ROOT_LOGGER_NAME = "jarvis"
LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 1_048_576
BACKUP_COUNT = 3

_configured: set[str] = set()


def resolve_level(level: str | int) -> int:
    """Convert a level name to a numeric level; unknown names raise."""
    if isinstance(level, int):
        return level
    candidate = level.strip().upper()
    numeric = logging.getLevelName(candidate)
    if not isinstance(numeric, int):
        raise ValueError(f"unknown logging level: {level!r}")
    return numeric


def setup_logging(
    config: LoggingConfig | None = None,
    *,
    level: str | None = None,
    root_name: str = ROOT_LOGGER_NAME,
    force: bool = False,
) -> logging.Logger:
    """Configure the `jarvis` logger tree and return the root logger."""
    config = config or LoggingConfig()
    logger = logging.getLogger(root_name)

    if root_name in _configured and not force:
        if level:
            logger.setLevel(resolve_level(level))
        return logger

    numeric_level = resolve_level(level or config.level)
    logger.setLevel(numeric_level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if config.console:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(formatter)
        console.setLevel(numeric_level)
        logger.addHandler(console)

    if config.to_file:
        file_handler = _build_file_handler(config, numeric_level, formatter, logger)
        if file_handler is not None:
            logger.addHandler(file_handler)

    _configured.add(root_name)
    logger.debug(
        "logging configured (level=%s, console=%s, file=%s)",
        logging.getLevelName(numeric_level),
        config.console,
        config.to_file,
    )
    return logger


def shutdown_logging() -> None:
    """Flush and close logging handlers (called on clean shutdown)."""
    logging.shutdown()
    _configured.clear()


def _build_file_handler(
    config: LoggingConfig,
    level: int,
    formatter: logging.Formatter,
    logger: logging.Logger,
) -> logging.Handler | None:
    try:
        path = config.log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        handler.setLevel(level)
        return handler
    except OSError as exc:
        logger.warning("file logging disabled, using console only: %s", exc)
        return None
