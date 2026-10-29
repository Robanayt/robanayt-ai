"""Voice in the surface layer: states are shown, nothing is invented."""

from __future__ import annotations

from pathlib import Path

from jarvis.surfaces.bridge import DialogBridge
from jarvis.surfaces.theme import DEFAULT_THEME
from jarvis.surfaces.ui import HolographicApp
from jarvis.surfaces.ui.settings import open_settings
from jarvis.voice.models import VoiceState
from jarvis.voice.router import VoiceRouter
from jarvis.voice.session import VoiceSession
from tests.orchestration.dialog_fakes import answer, make_dialog
from tests.surfaces.test_bridge import FakeDialog, WaitingDialog
from tests.voice.fakes import ScriptedSTT, ScriptedTTS, UnavailableSTT


def _bridge(tmp_path: Path, *, stt: object = None, transcript: str = "привет джарвис"):
    dialog, provider, memory, tools, _clock = make_dialog(tmp_path, [answer("ответ готов")])
    router = VoiceRouter(stt=(stt or ScriptedSTT(transcript),), tts=(ScriptedTTS(),))
    session = VoiceSession(router, dialog=dialog)
    bridge = DialogBridge(
        dialog, tools=tools, memory=memory, background=False, voice=session
    )
    return bridge, session, dialog


def test_spoken_text_becomes_an_ordinary_message(tmp_path: Path) -> None:
    bridge, _session, dialog = _bridge(tmp_path)

    assert bridge.start_voice() is True

    view = bridge.snapshot()
    assert [message.text for message in view.messages][-2:] == [
        "привет джарвис",
        "ответ готов",
    ]
    dialog.close()


def test_unheard_phrase_creates_no_message(tmp_path: Path) -> None:
    bridge, session, dialog = _bridge(tmp_path, stt=ScriptedSTT(""))

    bridge.start_voice()

    assert bridge.snapshot().messages == ()
    assert session.state == VoiceState.IDLE.value
    dialog.close()


def test_unavailable_provider_is_shown_as_unavailable(tmp_path: Path) -> None:
    bridge, session, dialog = _bridge(tmp_path, stt=UnavailableSTT("микрофон не найден"))

    bridge.start_voice()

    view = bridge.snapshot()
    assert view.voice is not None
    # The state is "unavailable", and the side that failed is named: with no
    # recogniser there is nothing to listen with.
    assert view.voice.state == VoiceState.UNAVAILABLE.value
    assert view.voice.stt_provider is None
    assert view.voice.tts_provider == "scripted_tts"
    assert bridge.snapshot().messages == ()
    assert "микрофон не найден" in (session.last_error or "")
    dialog.close()


def test_voice_report_shows_both_sides(tmp_path: Path) -> None:
    bridge, _session, dialog = _bridge(tmp_path)

    bridge.probe_voice()
    report = bridge.info()["voice"]

    assert report["stt"]["selected"] == "scripted_stt"
    assert report["tts"]["selected"] == "scripted_tts"
    assert report["state"] == VoiceState.IDLE.value
    dialog.close()


def test_bridge_without_voice_keeps_the_snapshot_shape() -> None:
    bridge = DialogBridge(FakeDialog(), background=False)

    view = bridge.snapshot()

    assert view.voice is None
    assert bridge.start_voice() is False
    assert bridge.speak("что-нибудь") is False
    assert bridge.cancel_voice() is False


def test_speaking_goes_through_the_voice_session(tmp_path: Path) -> None:
    bridge, session, dialog = _bridge(tmp_path)
    tts = session._router.select_tts().provider

    assert bridge.speak("озвучь") is True

    assert tts.spoken == ["озвучь"]
    dialog.close()


def test_cancel_without_an_operation_is_not_a_success(tmp_path: Path) -> None:
    bridge, _session, dialog = _bridge(tmp_path)

    assert bridge.cancel_voice() is False
    dialog.close()


# -- window -----------------------------------------------------------------


def test_window_shows_the_voice_state_and_disables_what_cannot_work(tmp_path: Path) -> None:
    bridge, _session, dialog = _bridge(tmp_path)
    app = HolographicApp(bridge, hidden=True)

    try:
        bridge.probe_voice()
        app._poll()

        assert "слушаю" in app._voice_var.get() or "микрофон" in app._voice_var.get()
        assert str(app._voice_speak_button["state"]) == "normal"
        assert app._voice_detail_var.get()

        top = open_settings(app.root, bridge, DEFAULT_THEME)
        assert top is not None
        _close(top)
    finally:
        app.close()
        dialog.close()


def test_window_marks_voice_unavailable_when_providers_are_missing(tmp_path: Path) -> None:
    bridge, _session, dialog = _bridge(tmp_path, stt=UnavailableSTT("нет микрофона"))
    app = HolographicApp(bridge, hidden=True)

    try:
        bridge.probe_voice()
        app._poll()

        # Nothing may be started: there is no provider that could listen.
        assert str(app._voice_listen_button["state"]) == "disabled"
        assert "нет микрофона" in app._voice_detail_var.get() or app._voice_detail_var.get()
    finally:
        app.close()
        dialog.close()


class _FakeVoiceSession:
    """Session stand-in: probes nothing, but owns a real-looking state."""

    def __init__(self, state: str = VoiceState.IDLE.value, *, busy: bool = False) -> None:
        self.state = state
        self.busy = busy
        self.closed = False

    def explain(self, *, probe: bool = False) -> dict[str, object]:
        # What a cached probe would leave behind: the state at probe time.
        return {
            "enabled": True,
            "state": VoiceState.IDLE.value,
            "privacy": "local_only",
            "allow_remote": False,
            "stt": {"selected": "scripted_stt", "reason": "выбран", "providers": []},
            "tts": {"selected": "scripted_tts", "reason": "выбран", "providers": []},
        }

    def transcribe(self, **kwargs):  # pragma: no cover - never reached here
        raise AssertionError("no capture in this test")

    def speak(self, text: str, **kwargs) -> bool:
        return bool(text.strip())

    def cancel(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


def test_the_shown_state_stays_live_while_the_probe_report_is_cached() -> None:
    """A cached probe must not freeze the state: `listening` has to show."""
    bridge = DialogBridge(
        FakeDialog(),
        background=False,
        voice=_FakeVoiceSession(VoiceState.LISTENING.value, busy=True),
    )

    view = bridge.snapshot()

    assert view.voice is not None
    assert view.voice.state == VoiceState.LISTENING.value
    assert view.voice.label == "слушаю"
    assert view.voice.active is True


def test_a_second_capture_does_not_start_while_voice_is_busy() -> None:
    """Two captures would fight for one microphone: the second is refused."""
    bridge = DialogBridge(
        FakeDialog(),
        background=False,
        voice=_FakeVoiceSession(VoiceState.LISTENING.value, busy=True),
    )

    assert bridge.start_voice() is False


def test_window_without_voice_shows_the_subsystem_is_absent() -> None:
    bridge = DialogBridge(WaitingDialog(), background=False)
    app = HolographicApp(bridge, hidden=True)

    try:
        app._poll()

        assert "не подключён" in app._voice_var.get()
        assert str(app._voice_listen_button["state"]) == "disabled"
        assert str(app._voice_speak_button["state"]) == "disabled"
    finally:
        app.close()


def _close(top) -> None:
    try:
        top.destroy()
    except Exception:  # noqa: BLE001 - Tk teardown races are not failures
        pass
