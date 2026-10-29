"""CLI voice diagnostics and explicit actions - never automatic."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from robanayt_ai import app as app_module
from robanayt_ai.app import EXIT_OK, EXIT_VOICE_ERROR, build_voice, main
from tests.voice.fakes import ScriptedRunner, command_ok

KIND_SPEAK = "kind=speak"
KIND_RECOGNIZE = "kind=recognize"


def _has_database(path: Path) -> bool:
    """True when a memory database appeared - i.e. memory was actually used."""
    return any(path.glob("**/*.sqlite3"))


def _inject_runner(monkeypatch, outputs: dict[str, object]) -> ScriptedRunner:
    """Route every helper call through a scripted stand-in (no PowerShell)."""
    runner = ScriptedRunner(outputs)
    original = build_voice
    monkeypatch.setattr(
        app_module, "build_voice", functools.partial(original, runner=runner)
    )
    return runner


# -- diagnostics ------------------------------------------------------------


def test_voice_status_reports_available_tts_and_blocked_stt(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    runner = _inject_runner(
        monkeypatch,
        {
            "kind=probe-voices": command_ok("VOICE|Microsoft Irina Desktop|ru-RU"),
            "kind=probe-recognizers": command_ok(""),
        },
    )

    assert main(["--voice"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "voice status" in out
    assert "local_only" in out
    assert "windows_sapi_tts" in out
    assert "недоступен" in out
    assert "InstalledRecognizers" in out
    # Diagnostics must never listen and never play: only probes run.
    assert all(KIND_SPEAK not in script for script in runner.scripts)
    assert all(KIND_RECOGNIZE not in script for script in runner.scripts)


def test_voice_status_creates_nothing_on_disk(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _inject_runner(
        monkeypatch,
        {
            "kind=probe-voices": command_ok("VOICE|voice|ru-RU"),
            "kind=probe-recognizers": command_ok(""),
        },
    )

    assert main(["--voice"]) == EXIT_OK

    assert not _has_database(tmp_path)


def test_voice_status_reports_a_disabled_subsystem(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_VOICE_ENABLED", "false")
    _inject_runner(monkeypatch, {"kind=probe-voices": command_ok("VOICE|voice|ru-RU")})

    assert main(["--voice"]) == EXIT_OK

    assert "disabled" in capsys.readouterr().out


# -- speaking ---------------------------------------------------------------


def test_voice_say_speaks_once_when_a_voice_exists(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    runner = _inject_runner(
        monkeypatch, {"kind=probe-voices": command_ok("VOICE|voice|ru-RU")}
    )

    assert main(["--voice-say", "привет"]) == EXIT_OK

    assert sum(KIND_SPEAK in script for script in runner.scripts) == 1
    assert "spoken successfully" in capsys.readouterr().out


def test_voice_say_reports_missing_voices(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    runner = _inject_runner(monkeypatch, {"kind=probe-voices": command_ok("")})

    assert main(["--voice-say", "привет"]) == EXIT_VOICE_ERROR

    assert any(KIND_SPEAK not in script for script in runner.scripts)
    assert "voice error" in capsys.readouterr().err


# -- listening --------------------------------------------------------------


def test_voice_listen_reports_no_recognizer(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    runner = _inject_runner(monkeypatch, {"kind=probe-recognizers": command_ok("")})

    assert main(["--voice-listen"]) == EXIT_VOICE_ERROR

    out = capsys.readouterr().err
    assert "распознавател" in out
    assert all(KIND_RECOGNIZE not in script for script in runner.scripts)


def test_voice_listen_with_silence_sends_nothing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    runner = _inject_runner(
        monkeypatch,
        {
            "kind=probe-recognizers": command_ok("RECOGNIZER|ru-RU|MS"),
            KIND_RECOGNIZE: command_ok("NO_SPEECH"),
        },
    )

    assert main(["--voice-listen"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "речь не распознана" in out
    # No dialogue, no database: nothing was understood, nothing stored.
    assert not _has_database(tmp_path)
    assert sum(KIND_RECOGNIZE in script for script in runner.scripts) == 1


def test_voice_listen_needs_a_dialogue_backend(tmp_path, monkeypatch, capsys) -> None:
    """A recognised phrase without a working AI layer is reported, not faked."""
    monkeypatch.chdir(tmp_path)
    _inject_runner(
        monkeypatch,
        {
            "kind=probe-recognizers": command_ok("RECOGNIZER|ru-RU|MS"),
            KIND_RECOGNIZE: command_ok("TEXT:привет\nCONFIDENCE:0.9"),
        },
    )

    assert main(["--voice-listen"]) == EXIT_VOICE_ERROR

    out = capsys.readouterr()
    assert "распознано: привет" in out.out
    assert "voice error" in out.err


def test_voice_privacy_default_is_local_only(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    voice = build_voice(app_module.load_config(None))

    assert voice.explain(probe=False)["privacy"] == "local_only"


def test_voice_is_not_probed_at_construction(tmp_path, monkeypatch) -> None:
    """Composing the subsystem must cost nothing and touch nothing."""
    monkeypatch.chdir(tmp_path)
    runner = ScriptedRunner({"kind=probe-voices": command_ok("VOICE|voice|ru-RU")})

    voice = build_voice(app_module.load_config(None), runner=runner)

    assert runner.scripts == []
    voice.close()


@pytest.mark.parametrize("argument", ["--voice", "--voice-say", "--voice-listen"])
def test_voice_commands_are_registered(argument: str) -> None:
    from robanayt_ai.app import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [argument] if argument != "--voice-say" else [argument, "текст"]
    )

    assert (argument == "--voice" and args.voice) or (
        argument == "--voice-say" and args.voice_say == "текст"
    ) or (argument == "--voice-listen" and args.voice_listen)


def test_workspace_has_no_database_after_voice_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _inject_runner(
        monkeypatch,
        {
            "kind=probe-voices": command_ok("VOICE|voice|ru-RU"),
            "kind=probe-recognizers": command_ok(""),
        },
    )

    main(["--voice"])

    # The logging directory may appear (as for any command); the memory
    # database must not - diagnostics touches no data.
    assert not _has_database(tmp_path)
