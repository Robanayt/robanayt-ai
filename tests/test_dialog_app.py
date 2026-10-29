"""Application-level dialog behaviour: startup sends nothing, turns are explicit."""

from __future__ import annotations

import json
import urllib.request

from robanayt_ai.app import EXIT_DIALOG_ERROR, EXIT_OK, main


class _FakeResponse:
    """Minimal stand-in for an HTTP response (no sockets involved)."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _ollama_reply(content: str) -> bytes:
    return json.dumps(
        {
            "model": "test-model",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
        }
    ).encode("utf-8")


def _block_network(monkeypatch) -> list[str]:
    calls: list[str] = []

    def _boom(*args: object, **kwargs: object) -> None:
        calls.append("called")
        raise AssertionError("network access attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    return calls


def test_startup_registers_the_dialog_layer(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _block_network(monkeypatch)

    assert main([]) == EXIT_OK
    output = capsys.readouterr().out

    assert "dialog:" in output
    assert "tool call limit: 3 per message" in output
    assert "turns:          nothing" in output
    assert calls == []


def test_startup_creates_no_memory_database(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main([]) == EXIT_OK
    assert not (tmp_path / "runtime" / "memory" / "jarvis.db").exists()


def test_dialog_send_without_a_model_fails_cleanly(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _block_network(monkeypatch)

    assert main(["--dialog-send", "hello"]) == EXIT_DIALOG_ERROR

    out = capsys.readouterr().out
    assert "status:       ai_error" in out
    assert calls == []  # a failed turn never falls back to the network
    # An explicit dialog turn is first use: the database is created lazily,
    # exactly like the --memory-* commands, and never at startup.
    assert (tmp_path / "runtime" / "memory" / "jarvis.db").exists()


def test_dialog_send_runs_one_turn(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        '[ai]\ndefault_model = "test-model"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: _FakeResponse(_ollama_reply("Hi there"))
    )

    assert main(["--dialog-send", "hello"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "status:       success" in out
    assert "Hi there" in out
    assert "tool calls:   0" in out
    assert (tmp_path / "runtime" / "memory" / "jarvis.db").exists()


def test_dialog_continues_a_conversation(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        '[ai]\ndefault_model = "test-model"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: _FakeResponse(_ollama_reply("reply"))
    )

    assert main(["--dialog-send", "first"]) == EXIT_OK
    first = capsys.readouterr().out
    conversation_id = [
        line.split("conversation:")[1].strip()
        for line in first.splitlines()
        if "conversation:" in line
    ][0]

    assert main(["--dialog-send", "second", "--dialog-conversation", conversation_id]) == EXIT_OK

    second = capsys.readouterr().out
    assert "status:       success" in second
    assert conversation_id in second


def test_dialog_approve_without_pending_fails_cleanly(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main(["--dialog-approve", "dlg_missing"]) == EXIT_DIALOG_ERROR
    assert "pending_expired" in capsys.readouterr().out


def test_dialog_pending_is_empty_on_a_fresh_start(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main(["--dialog-pending"]) == EXIT_OK
    assert "(none)" in capsys.readouterr().out


def test_dialog_send_rejects_an_empty_message(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main(["--dialog-send", "   "]) == EXIT_DIALOG_ERROR
    assert "dialog error" in capsys.readouterr().err
