"""Application integration: startup stays lazy, memory CLI is explicit."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from jarvis.app import EXIT_MEMORY_ERROR, EXIT_OK, main


def _block_network(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)


def _db_path(tmp_path):
    return tmp_path / "runtime" / "memory" / "jarvis.db"


def test_startup_does_not_create_the_database(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main([]) == EXIT_OK
    assert _db_path(tmp_path).exists() is False

    out = capsys.readouterr().out
    assert "memory:" in out
    assert "not opened (lazy" in out


def test_startup_reports_disabled_memory(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        "[memory]\nenabled = false\n", encoding="utf-8"
    )

    assert main([]) == EXIT_OK
    assert "disabled by configuration" in capsys.readouterr().out
    assert _db_path(tmp_path).exists() is False


def test_memory_status_is_explicit_and_creates_store(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _block_network(monkeypatch)

    assert main(["--memory-status"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "schema version" in out
    assert _db_path(tmp_path).exists() is True


def test_memory_list_shows_metadata_only(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--memory-list"]) == EXIT_OK
    assert "(none)" in capsys.readouterr().out


def test_memory_show_unknown_id_fails_cleanly(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--memory-show", "conv_missing"]) == EXIT_MEMORY_ERROR
    assert "memory error" in capsys.readouterr().err


def test_memory_round_trip_through_cli(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    # Create data through the CLI-visible path, then export and read it back.
    assert main(["--memory-status"]) == EXIT_OK
    capsys.readouterr()

    from jarvis.core.config import MemoryConfig
    from jarvis.memory.service import MemoryService

    service = MemoryService(MemoryConfig(directory="runtime/memory"))
    conversation = service.create_conversation("cli trip", now=1.0)
    service.add_message(conversation.id, "user", "cli visible text")
    service.close()

    assert main(["--memory-list"]) == EXIT_OK
    assert "cli trip" in capsys.readouterr().out

    export_path = tmp_path / "exported.json"
    assert main(["--memory-export", conversation.id, "--memory-out", str(export_path)]) == EXIT_OK
    assert export_path.is_file()

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["messages"][0]["content"] == "cli visible text"

    # A second export to the same path is refused without --memory-out overwrite.
    assert main(["--memory-export", conversation.id, "--memory-out", str(export_path)]) == EXIT_MEMORY_ERROR


def test_memory_search_prints_snippets(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    from jarvis.core.config import MemoryConfig
    from jarvis.memory.service import MemoryService

    service = MemoryService(MemoryConfig(directory="runtime/memory"))
    conversation = service.create_conversation("searchable", now=1.0)
    service.add_message(conversation.id, "user", "find the hidden keyword")
    service.close()

    assert main(["--memory-search", "hidden"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "hidden" in out
    assert "[message]" in out


def test_memory_clear_requires_confirmation(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--memory-clear"]) == EXIT_MEMORY_ERROR
    assert "refusing" in capsys.readouterr().err


def test_disabled_memory_cli_fails_clearly(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        "[memory]\nenabled = false\n", encoding="utf-8"
    )

    assert main(["--memory-status"]) == EXIT_MEMORY_ERROR
    assert "disabled" in capsys.readouterr().err


def test_memory_commands_do_not_touch_the_network(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def _boom(*args, **kwargs):
        raise urllib.error.URLError("network disabled")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    assert main(["--memory-status"]) == EXIT_OK
    assert main(["--memory-list"]) == EXIT_OK
