"""Phase 1 smoke tests: the core-wired skeleton starts and exits cleanly."""

from __future__ import annotations

import robanayt_ai
from robanayt_ai.app import EXIT_OK, STATUS_NO_COMPONENTS, build_startup_message, main


def test_package_exposes_metadata() -> None:
    assert robanayt_ai.__version__ == "0.5.0"
    assert robanayt_ai.__phase__ == "phase-8"


def test_startup_message_is_clear() -> None:
    message = build_startup_message()
    assert "Robanayt AI" in message
    assert STATUS_NO_COMPONENTS in message


def test_main_prints_status_and_exits_cleanly(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == EXIT_OK
    assert STATUS_NO_COMPONENTS in captured.out
    assert "Traceback" not in captured.err
