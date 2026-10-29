"""Logging: stdlib only, idempotent setup, usable diagnostics."""

from __future__ import annotations

import logging

import pytest

from jarvis.core.config import LoggingConfig
from jarvis.core.log_config import resolve_level, setup_logging, shutdown_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    yield
    shutdown_logging()


def test_resolve_level_accepts_names() -> None:
    assert resolve_level("debug") == logging.DEBUG
    assert resolve_level("INFO") == logging.INFO
    assert resolve_level(logging.WARNING) == logging.WARNING


def test_resolve_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown logging level"):
        resolve_level("LOUD")


def test_console_handler_configured() -> None:
    logger = setup_logging(LoggingConfig(level="DEBUG", to_file=False), force=True)

    assert logger.name == "jarvis"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1


def test_file_handler_writes_records(tmp_path) -> None:
    logger = setup_logging(
        LoggingConfig(level="INFO", directory=str(tmp_path), filename="t.log"),
        force=True,
    )
    logger.info("hello-from-jarvis")

    for handler in logger.handlers:
        handler.flush()

    content = (tmp_path / "t.log").read_text(encoding="utf-8")
    assert "hello-from-jarvis" in content
    assert "INFO" in content


def test_setup_is_idempotent_without_force() -> None:
    setup_logging(LoggingConfig(level="INFO", to_file=False), force=True)
    before = len(logging.getLogger("jarvis").handlers)

    setup_logging(LoggingConfig(level="INFO", to_file=False))

    assert len(logging.getLogger("jarvis").handlers) == before


def test_level_override_without_reconfiguration() -> None:
    logger = setup_logging(LoggingConfig(level="INFO", to_file=False), force=True)

    setup_logging(LoggingConfig(level="INFO", to_file=False), level="DEBUG")

    assert logger.level == logging.DEBUG


def test_invalid_level_raises() -> None:
    with pytest.raises(ValueError):
        setup_logging(LoggingConfig(level="LOUD"), force=True)


def test_shutdown_is_safe() -> None:
    setup_logging(LoggingConfig(level="INFO", to_file=False), force=True)

    shutdown_logging()

    # Reconfiguration after shutdown still works.
    logger = setup_logging(LoggingConfig(level="INFO", to_file=False), force=True)
    assert logger.handlers
