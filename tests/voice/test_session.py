"""Voice session lifecycle: states, cancellation, privacy and honesty."""

from __future__ import annotations

import threading
import time

import pytest

from jarvis.core.bus import EventBus
from jarvis.orchestration.turn import TurnCancellation
from jarvis.voice.errors import SpeechError, TranscriptionError
from jarvis.voice.models import VoiceState
from jarvis.voice.router import LOCAL_ONLY, VoiceRouter
from jarvis.voice.session import VOICE_STATE_CHANGED, VoiceSession
from tests.voice.fakes import (
    RecordingDialog,
    ScriptedSTT,
    ScriptedTTS,
    UnavailableSTT,
)


def _session(stt: object = None, tts: object = None, **kwargs) -> tuple[VoiceSession, EventBus, list]:
    bus = EventBus()
    states: list[str] = []
    bus.subscribe(VOICE_STATE_CHANGED, lambda event: states.append(event.get("state")))
    privacy = kwargs.pop("privacy", LOCAL_ONLY)
    router = VoiceRouter(
        stt=(stt if stt is not None else ScriptedSTT(),),
        tts=(tts if tts is not None else ScriptedTTS(),),
        privacy=privacy,
        allow_remote=kwargs.pop("allow_remote", False),
    )
    session = VoiceSession(router, bus=bus, privacy=privacy, **kwargs)
    return session, bus, states


# -- listening --------------------------------------------------------------


def test_a_heard_phrase_becomes_one_dialogue_turn() -> None:
    dialog = RecordingDialog("ответ")
    session, _bus, states = _session(dialog=dialog)

    turn = session.listen()

    assert turn.transcript == "привет джарвис"
    assert turn.answer == "ответ"
    assert len(dialog.calls) == 1
    assert dialog.calls[0]["message"] == "привет джарвис"
    assert states[:4] == [
        VoiceState.LISTENING.value,
        VoiceState.TRANSCRIBING.value,
        VoiceState.THINKING.value,
        VoiceState.IDLE.value,
    ]


def test_an_empty_transcript_is_never_sent_to_the_dialogue() -> None:
    dialog = RecordingDialog()
    session, _bus, _states = _session(stt=ScriptedSTT(""), dialog=dialog)

    turn = session.listen()

    assert turn.transcript_empty is True
    assert dialog.calls == []
    assert session.state == VoiceState.IDLE.value


def test_transcription_failure_is_reported_not_hidden() -> None:
    dialog = RecordingDialog()
    session, _bus, _states = _session(stt=ScriptedSTT(error=TranscriptionError("устройство занято")), dialog=dialog)

    turn = session.listen()

    assert "устройство занято" in turn.error
    assert session.state == VoiceState.ERROR.value
    assert dialog.calls == []


def test_unavailable_microphone_is_reported_as_unavailable() -> None:
    dialog = RecordingDialog()
    session, _bus, _states = _session(stt=UnavailableSTT("микрофон не найден"), dialog=dialog)

    turn = session.listen()

    assert "микрофон не найден" in turn.error
    assert session.state == VoiceState.UNAVAILABLE.value
    assert dialog.calls == []


def test_disabled_session_refuses_to_listen() -> None:
    session, _bus, _states = _session(enabled=False)

    turn = session.listen()

    assert "отключен" in turn.error
    assert session.state == VoiceState.UNAVAILABLE.value


# -- cancellation -----------------------------------------------------------


class BlockingSTT:
    """Listens until the cooperative flag is set, then stops cooperatively."""

    def __init__(self) -> None:
        self.id = "blocking_stt"
        self.kind = "local"
        self.stopped = threading.Event()
        self.calls: list[dict] = []

    def status(self, *, probe: bool = False):
        from jarvis.voice.models import VoiceProviderStatus

        return VoiceProviderStatus(provider=self.id, configured=True, available=True)

    def describe(self):
        return {"provider": self.id}

    @property
    def capabilities(self):
        from jarvis.voice.models import VoiceCapabilities

        return VoiceCapabilities(local=True, streaming=False, cancel_supported=True)

    def transcribe(self, *, timeout_seconds: float, language: str = "", cancellation=None):
        self.calls.append({"timeout": timeout_seconds})
        deadline = time.time() + min(timeout_seconds, 3.0)
        while time.time() < deadline:
            if cancellation is not None and cancellation.requested:
                from jarvis.voice.errors import VoiceCancelledError

                self.stopped.set()
                raise VoiceCancelledError("распознавание отменено")
            time.sleep(0.01)
        from jarvis.voice.errors import VoiceCancelledError

        raise VoiceCancelledError("время вышло")


def test_cancel_stops_the_listening_and_says_it_was_cancelled() -> None:
    dialog = RecordingDialog()
    provider = BlockingSTT()
    session, _bus, states = _session(stt=provider, dialog=dialog)
    outcome: dict[str, object] = {}

    def worker() -> None:
        outcome["turn"] = session.listen()

    thread = threading.Thread(target=worker)
    thread.start()
    while not provider.calls:
        time.sleep(0.01)
    assert session.cancel() is True
    thread.join(timeout=5.0)

    turn = outcome["turn"]
    assert turn.state == VoiceState.CANCELLED.value
    assert provider.stopped.is_set()
    assert dialog.calls == []
    assert VoiceState.CANCELLING.value in states


def test_cancelling_nothing_is_not_a_success() -> None:
    session, _bus, _states = _session()

    assert session.cancel() is False
    assert session.busy is False


def test_a_cancelled_turn_is_never_spoken_aloud() -> None:
    """Cancellation must not leave side effects after the requested stop."""
    tts = ScriptedTTS()

    class CancellingDialog:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def send(self, message: str, **kwargs: object):
            self.calls.append({"message": message, **kwargs})
            kwargs["cancellation"].cancel()
            return _CancelledResult()

    dialog = CancellingDialog()
    session, _bus, _states = _session(tts=tts, dialog=dialog, speak_responses=True)

    turn = session.listen()

    assert turn.state == VoiceState.CANCELLED.value
    assert tts.spoken == []
    assert dialog.calls[0]["message"] == "привет джарвис"


class _StatusValue:
    def __init__(self, value: str) -> None:
        self.value = value


class _CancelledResult:
    answer = "не должно звучать"
    error = ""
    status = _StatusValue("cancelled")
    tool_calls: tuple = ()
    conversation_id = None


# -- speaking ---------------------------------------------------------------


def test_speak_sends_the_text_to_the_tts_provider() -> None:
    tts = ScriptedTTS()
    session, _bus, states = _session(tts=tts)

    assert session.speak("озвучь это") is True
    assert tts.spoken == ["озвучь это"]
    assert VoiceState.SPEAKING.value in states


def test_speak_ignores_empty_text() -> None:
    tts = ScriptedTTS()
    session, _bus, _states = _session(tts=tts)

    assert session.speak("   ") is False
    assert tts.spoken == []


def test_speak_reports_an_unavailable_provider() -> None:
    tts = ScriptedTTS(available=False, message="голос не установлен")
    session, _bus, _states = _session(tts=tts)

    assert session.speak("привет") is False
    assert session.state == VoiceState.UNAVAILABLE.value
    assert "голос не установлен" in (session.last_error or "")


def test_speak_reports_a_synthesis_error() -> None:
    session, _bus, _states = _session(tts=ScriptedTTS(error=SpeechError("сбой синтеза")))

    assert session.speak("привет") is False
    assert session.state == VoiceState.ERROR.value
    assert "сбой синтеза" in (session.last_error or "")


def test_answers_are_spoken_only_when_asked() -> None:
    tts = ScriptedTTS()
    dialog = RecordingDialog("озвученный ответ")
    session, _bus, _states = _session(tts=tts, dialog=dialog, speak_responses=True)

    turn = session.listen()

    assert turn.spoken is True
    assert tts.spoken == ["озвученный ответ"]

    quiet_tts = ScriptedTTS(provider_id="quiet_tts")
    quiet, _bus2, _states2 = _session(tts=quiet_tts, dialog=dialog)
    quiet.listen()
    assert quiet_tts.spoken == []


# -- privacy ----------------------------------------------------------------


def test_privacy_mode_reaches_the_dialogue_layer() -> None:
    dialog = RecordingDialog()
    session, _bus, _states = _session(dialog=dialog, privacy=LOCAL_ONLY)

    session.listen()

    assert dialog.calls[0]["privacy"] == LOCAL_ONLY


def test_explain_describes_the_subsystem_without_probing() -> None:
    session, _bus, _states = _session(privacy=LOCAL_ONLY)

    report = session.explain(probe=False)

    assert report["enabled"] is True
    assert report["privacy"] == LOCAL_ONLY
    assert report["state"] == VoiceState.IDLE.value
    assert set(report["stt"]) >= {"selected", "reason", "rejected", "providers"}


def test_explain_marks_a_disabled_subsystem() -> None:
    session, _bus, _states = _session(enabled=False)

    report = session.explain(probe=False)

    assert report["enabled"] is False
    assert report["state"] == VoiceState.UNAVAILABLE.value


def test_close_leaves_no_operation_running() -> None:
    session, _bus, _states = _session()

    session.close()

    assert session.state == VoiceState.IDLE.value
    assert session.snapshot().state == VoiceState.IDLE.value


class _ClosingSTT(ScriptedSTT):
    """Provider that closes the session while it would still be listening."""

    def __init__(self) -> None:
        super().__init__()
        self.state_after_close: str | None = None

    def bind(self, session: VoiceSession) -> None:
        self._session = session

    def transcribe(self, **kwargs):
        self._session.close()
        self.state_after_close = self._session.state
        return super().transcribe(**kwargs)


def test_close_stops_claiming_the_microphone_is_open() -> None:
    """A closed session must not keep reporting `listening`."""
    provider = _ClosingSTT()
    session = VoiceSession(VoiceRouter(stt=(provider,), tts=(ScriptedTTS(),)))
    provider.bind(session)

    turn = session.transcribe()

    assert provider.state_after_close == VoiceState.CANCELLED.value
    assert session.state == VoiceState.CANCELLED.value
    assert turn.state == VoiceState.CANCELLED.value
