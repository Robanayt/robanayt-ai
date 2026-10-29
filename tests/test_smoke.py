"""Phase 1 smoke tests: the core-wired skeleton starts and exits cleanly."""

from __future__ import annotations

import jarvis
from jarvis.app import EXIT_OK, STATUS_NO_COMPONENTS, build_startup_message, main


def test_package_exposes_metadata() -> None:
    assert jarvis.__version__ == "0.5.0"
    assert jarvis.__phase__ == "phase-8"


def test_startup_message_is_clear() -> None:
    message = build_startup_message()
    assert "JARVIS_AI" in message
    assert STATUS_NO_COMPONENTS in message


def test_main_prints_status_and_exits_cleanly(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == EXIT_OK
    assert STATUS_NO_COMPONENTS in captured.out
    assert "Traceback" not in captured.err
