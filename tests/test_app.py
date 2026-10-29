"""Application integration: startup, shutdown, exit codes, log artifacts."""

from __future__ import annotations

import json
import os

import pytest
from dataclasses import replace

from robanayt_ai.ai.errors import ProviderUnavailableError
from robanayt_ai.app import (
    EXIT_ALREADY_RUNNING,
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    STATUS_NO_COMPONENTS,
    build_ai_service,
    build_parser,
    build_tools,
    main,
    run,
)
from robanayt_ai.core.config import load_config


def test_main_starts_and_exits_cleanly(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main([]) == EXIT_OK

    captured = capsys.readouterr()
    assert STATUS_NO_COMPONENTS in captured.out
    assert "Traceback" not in captured.err


def test_main_reports_run_mode(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--run-mode", "diagnostic"]) == EXIT_OK

    assert "run mode: diagnostic" in capsys.readouterr().out


def test_main_rejects_unknown_run_mode(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--run-mode", "turbo"]) == EXIT_CONFIG_ERROR

    assert "configuration error" in capsys.readouterr().err


def test_main_writes_log_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main([]) == EXIT_OK

    log_file = tmp_path / "runtime" / "logs" / "jarvis.log"
    assert log_file.is_file()
    assert "core ready" in log_file.read_text(encoding="utf-8")


def test_main_refuses_second_instance(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    lock_path = tmp_path / "runtime" / "jarvis.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

    assert main([]) == EXIT_ALREADY_RUNNING

    captured = capsys.readouterr()
    assert "already running" in captured.err.lower()
    # The other instance's lock is never removed.
    assert lock_path.is_file()


def test_no_single_instance_flag_skips_the_lock(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    lock_path = tmp_path / "runtime" / "jarvis.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

    assert main(["--no-single-instance"]) == EXIT_OK


def test_repeated_runs_release_the_lock(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert run([]) == EXIT_OK

    lock_path = tmp_path / "runtime" / "jarvis.lock"
    assert not lock_path.exists()


def test_run_mode_disables_the_tool_executor(tmp_path, monkeypatch) -> None:
    """The mode boundary reaches the executor, not just the startup banner."""
    monkeypatch.chdir(tmp_path)
    config = load_config(None)

    allowed = build_tools(config, allow_tools=True)
    blocked = build_tools(config, allow_tools=False)

    assert allowed.executor.enabled is True
    assert blocked.executor.enabled is False
    # The registry stays readable: describing tools is not a side effect.
    assert len(blocked.registry) == len(allowed.registry) > 0
    allowed.close()
    blocked.close()


def test_run_mode_blocks_remote_providers(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "test-key-not-real")
    config = load_config(None)
    ai = replace(
        config.ai,
        openai_base_url="https://api.example.test/v1",
        openai_model="gpt-test",
        allow_remote=True,
    )
    scoped = replace(config, ai=ai)

    blocked = build_ai_service(scoped, allow_network=False)
    permitted = build_ai_service(scoped, allow_network=True)

    blocked_messages = " ".join(status.message or "" for status in blocked.status())
    permitted_messages = " ".join(status.message or "" for status in permitted.status())

    assert "remote calls disabled" in blocked_messages
    assert "remote calls disabled" not in permitted_messages


def test_startup_message_reports_mode_boundaries(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--run-mode", "safe"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "boundaries: network=blocked, tools=blocked" in out


def test_ui_flag_is_recognised() -> None:
    assert build_parser().parse_args(["--ui"]).ui is True
    assert build_parser().parse_args([]).ui is False


def test_ai_routing_report_is_offline(tmp_path, monkeypatch, capsys) -> None:
    """The routing explanation is diagnostics: no provider is contacted."""
    monkeypatch.chdir(tmp_path)

    assert main(["--ai-routing"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "policy:" in out
    assert "active:" in out
    assert "why:" in out
    assert "network: not contacted" in out


def test_blocked_network_removes_remote_providers_from_routing(tmp_path, monkeypatch) -> None:
    """Regression: remote was selectable even when the mode blocked it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "test-key-not-real")
    config = load_config(None)
    ai = replace(
        config.ai,
        default_model="",
        ollama_model="",
        openai_base_url="https://api.example.test/v1",
        openai_model="gpt-test",
        allow_remote=True,
        routing_policy="cloud_first",
    )
    scoped = replace(config, ai=ai)

    blocked = build_ai_service(scoped, allow_network=False)
    permitted = build_ai_service(scoped, allow_network=True)

    assert blocked.router.allow_remote is False
    with pytest.raises(ProviderUnavailableError, match="allow_remote=false"):
        blocked.router.select()
    assert permitted.router.select().provider.name == "openai_compatible"


def test_ui_reports_a_missing_toolkit(tmp_path, monkeypatch, capsys) -> None:
    """Without tkinter the interface is reported, not faked."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("robanayt_ai.app.TK_AVAILABLE", False)

    assert main(["--ui"]) == EXIT_CONFIG_ERROR

    assert "interface unavailable" in capsys.readouterr().err
