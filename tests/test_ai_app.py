"""Application-level AI integration: startup is offline, probes are explicit."""

from __future__ import annotations

import urllib.error
import urllib.request

from jarvis.app import EXIT_AI_ERROR, EXIT_OK, main


def _block_network(monkeypatch) -> None:
    """Fail loudly if anything tries to open a socket during startup."""

    def _boom(*args, **kwargs):
        raise AssertionError("network access attempted during startup")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)


def test_startup_never_contacts_a_provider(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main([]) == EXIT_OK

    captured = capsys.readouterr()
    assert "AI:" in captured.out
    assert "service initialized" in captured.out
    # Phase 7: the banner shows the routing decision, not just a name.
    assert "routing policy: local_first" in captured.out
    assert "active:" in captured.out
    assert "why:" in captured.out
    assert "network: not contacted" in captured.out


def test_diagnostic_startup_is_also_offline(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main(["--run-mode", "diagnostic"]) == EXIT_OK
    assert "network: not contacted" in capsys.readouterr().out


def test_ai_status_without_model_reports_configuration(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main(["--ai-status"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "ollama" in out
    assert "configured=False" in out


def test_ai_status_detects_unavailable_ollama(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        '[ai]\ndefault_model = "probe-model"\n', encoding="utf-8"
    )

    def _offline(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _offline)

    assert main(["--ai-status"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "available=False" in out
    assert "probe failed" in out


def test_ai_generate_without_model_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main(["--ai-generate", "hello"]) == EXIT_AI_ERROR

    err = capsys.readouterr().err
    assert "AI error" in err
    assert "no model configured" in err


def test_ai_generate_rejects_unknown_provider(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        '[ai]\ndefault_model = "m"\n', encoding="utf-8"
    )
    _block_network(monkeypatch)

    assert main(["--ai-generate", "hi", "--ai-provider", "nope"]) == EXIT_AI_ERROR

    assert "unknown provider" in capsys.readouterr().err


def test_only_flagged_runs_may_reach_the_network(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        '[ai]\ndefault_model = "probe-model"\n', encoding="utf-8"
    )
    calls: list[str] = []

    def _record(*args, **kwargs):
        calls.append("called")
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", _record)

    # Plain startup with a model configured still makes no request.
    assert main([]) == EXIT_OK
    assert calls == []

    # Explicit probe: the network layer is reached and fails gracefully.
    assert main(["--ai-status"]) == EXIT_OK
    assert calls == ["called"]
