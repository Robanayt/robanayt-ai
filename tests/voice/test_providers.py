"""Windows SAPI providers: availability is proven, never assumed."""

from __future__ import annotations

import pytest

from jarvis.voice import VoiceCancelledError, VoiceUnavailableError
from jarvis.voice.providers.sapi import (
    SapiSpeechRecognizer,
    SapiSpeechSynthesizer,
    _b64,
)
from tests.voice.fakes import ScriptedRunner, command_fail, command_ok


def _synth(runner: ScriptedRunner) -> SapiSpeechSynthesizer:
    return SapiSpeechSynthesizer(runner=runner)


def _recognizer(runner: ScriptedRunner) -> SapiSpeechRecognizer:
    return SapiSpeechRecognizer(runner=runner)


KIND_TTS_PROBE = "kind=probe-voices"
KIND_STT_PROBE = "kind=probe-recognizers"
KIND_SPEAK = "kind=speak"
KIND_RECOGNIZE = "kind=recognize"


# -- availability -----------------------------------------------------------


def test_tts_is_available_when_windows_really_lists_voices() -> None:
    runner = ScriptedRunner(
        {KIND_TTS_PROBE: command_ok("VOICE|Microsoft Irina Desktop|ru-RU")}
    )

    status = _synth(runner).status(probe=True)

    assert status.available is True
    assert status.metadata["voices"] == ["Microsoft Irina Desktop"]
    assert "1" in status.message


def test_tts_reports_no_windows_voices() -> None:
    runner = ScriptedRunner({KIND_TTS_PROBE: command_ok("")})

    status = _synth(runner).status(probe=True)

    assert status.available is False
    assert "не установлены" in status.message


def test_status_without_probe_says_it_was_not_checked() -> None:
    runner = ScriptedRunner({KIND_TTS_PROBE: command_ok("VOICE|voice|ru-RU")})

    status = _synth(runner).status()

    assert status.available is False
    assert "не проверена" in status.message


def test_probe_failure_is_reported_not_raised() -> None:
    runner = ScriptedRunner({KIND_TTS_PROBE: command_fail()})

    status = _synth(runner).status(probe=True)

    assert status.available is False
    assert "сбой" in status.message or "PowerShell" in status.message


def test_stt_reports_missing_recognizer_explicitly() -> None:
    runner = ScriptedRunner({KIND_STT_PROBE: command_ok("")})

    status = _recognizer(runner).status(probe=True)

    assert status.available is False
    # The reason names the real thing that is missing.
    assert "InstalledRecognizers" in status.message


def test_stt_is_available_when_a_recognizer_exists() -> None:
    runner = ScriptedRunner({KIND_STT_PROBE: command_ok("RECOGNIZER|ru-RU|MS")})

    status = _recognizer(runner).status(probe=True)

    assert status.available is True
    assert status.metadata["recognizers"] == ["ru-RU"]


# -- speaking ---------------------------------------------------------------


def test_speak_sends_the_text_as_base64_payload_only() -> None:
    runner = ScriptedRunner(
        {KIND_TTS_PROBE: command_ok("VOICE|voice|ru-RU"), KIND_SPEAK: command_ok()}
    )
    synthesizer = _synth(runner)

    assert synthesizer.speak("привет; Remove-Item C:\\") is None

    script = runner.scripts[-1]
    assert KIND_SPEAK in script
    # The raw wording never appears as code: only a base64 payload does.
    assert "Remove-Item" not in script
    assert "привет" not in script


def test_speak_ignores_empty_text_instead_of_playing_silence() -> None:
    runner = ScriptedRunner({KIND_TTS_PROBE: command_ok("VOICE|voice|ru-RU")})
    synthesizer = _synth(runner)
    status_before = synthesizer.status(probe=True)

    assert synthesizer.speak("   ") is None
    # Only the probe ran: no speech was attempted.
    assert len(runner.scripts) == 1
    assert status_before.available is True


def test_speak_refuses_text_longer_than_the_limit() -> None:
    runner = ScriptedRunner({KIND_TTS_PROBE: command_ok("VOICE|voice|ru-RU")})

    with pytest.raises(Exception, match="лимит"):
        _synth(runner).speak("а" * 2500)


def test_speak_fails_loudly_when_no_voice_exists() -> None:
    runner = ScriptedRunner({KIND_TTS_PROBE: command_ok("")})

    with pytest.raises(VoiceUnavailableError):
        _synth(runner).speak("привет")


def test_speak_reports_backend_timeout() -> None:
    from jarvis.voice.providers.powershell import CommandResult

    runner = ScriptedRunner(
        {
            KIND_TTS_PROBE: command_ok("VOICE|voice|ru-RU"),
            KIND_SPEAK: CommandResult(False, -1, error="время истекло", timed_out=True),
        }
    )

    with pytest.raises(Exception, match="вовремя"):
        _synth(runner).speak("привет")


def test_speak_reports_cancellation() -> None:
    from jarvis.orchestration.turn import TurnCancellation

    runner = ScriptedRunner({KIND_TTS_PROBE: command_ok("VOICE|voice|ru-RU")})
    synthesizer = _synth(runner)
    cancellation = TurnCancellation()
    cancellation.cancel()

    with pytest.raises(VoiceCancelledError, match="отмен"):
        synthesizer.speak("привет", cancellation=cancellation)


def test_transcribe_reports_cancellation() -> None:
    from jarvis.orchestration.turn import TurnCancellation

    runner = ScriptedRunner({KIND_STT_PROBE: command_ok("RECOGNIZER|ru-RU|MS")})
    recognizer = _recognizer(runner)
    cancellation = TurnCancellation()
    cancellation.cancel()

    with pytest.raises(VoiceCancelledError, match="отмен"):
        recognizer.transcribe(timeout_seconds=2.0, cancellation=cancellation)


# -- transcribing -----------------------------------------------------------


def test_recognize_parses_the_recognized_text() -> None:
    runner = ScriptedRunner(
        {
            KIND_STT_PROBE: command_ok("RECOGNIZER|ru-RU|MS"),
            KIND_RECOGNIZE: command_ok("TEXT:привет джарвис\nCONFIDENCE:0.87"),
        }
    )

    transcription = _recognizer(runner).transcribe(timeout_seconds=5.0)

    assert transcription.text == "привет джарвис"
    assert transcription.provider == _recognizer(runner).id
    assert transcription.confidence == pytest.approx(0.87)


def test_unrecognized_speech_gives_an_empty_transcription() -> None:
    runner = ScriptedRunner(
        {
            KIND_STT_PROBE: command_ok("RECOGNIZER|ru-RU|MS"),
            KIND_RECOGNIZE: command_fail("NO_SPEECH"),
        }
    )
    recognizer = _recognizer(runner)
    recognizer.status(probe=True)

    transcription = recognizer.transcribe(timeout_seconds=2.0)

    # No speech is not an exception: it is information.
    assert transcription.empty is True


def test_transcribe_refuses_when_no_recognizer_is_installed() -> None:
    runner = ScriptedRunner({KIND_STT_PROBE: command_ok("")})

    with pytest.raises(VoiceUnavailableError, match="распознавател"):
        _recognizer(runner).transcribe(timeout_seconds=2.0)


def test_transcribe_refuses_an_unsupported_language() -> None:
    runner = ScriptedRunner({KIND_STT_PROBE: command_ok("RECOGNIZER|ru-RU|MS")})

    with pytest.raises(VoiceUnavailableError, match="языка"):
        _recognizer(runner).transcribe(timeout_seconds=2.0, language="de-DE")


def test_transcribe_passes_only_base64_and_a_timeout() -> None:
    runner = ScriptedRunner(
        {
            KIND_STT_PROBE: command_ok("RECOGNIZER|ru-RU|MS"),
            KIND_RECOGNIZE: command_ok("TEXT:тест"),
        }
    )
    recognizer = _recognizer(runner)
    recognizer.status(probe=True)

    recognizer.transcribe(timeout_seconds=4.0, language="ru")

    script = runner.scripts[-1]
    assert "FromSeconds(4.0)" in script
    assert _b64("ru") in script  # the language travels encoded, never as code


def test_provider_metadata_never_carries_secrets() -> None:
    runner = ScriptedRunner({KIND_TTS_PROBE: command_ok("VOICE|voice|ru-RU")})
    synthesizer = _synth(runner)
    synthesizer.status(probe=True)

    blob = str(synthesizer.describe()).lower()

    assert "api_key" not in blob
    assert "secret" not in blob
    assert synthesizer.capabilities.local is True
    assert synthesizer.capabilities.streaming is False

