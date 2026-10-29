"""Voice session: the lifecycle of one spoken interaction.

Design rules:

- Voice creates **no second AI pipeline**: a transcript goes into the existing
  `DialogService`, exactly like typed text, with the same permissions, the same
  tool policy and the same refusal behaviour.
- The raw microphone stream never leaves this layer: no audio is stored, no
  audio is logged, and nothing is written to long-term memory beyond what a
  normal text turn would produce.
- States are real. `listening` means something is listening; `unavailable`
  means a probe said so, with a reason.
- Cancellation is cooperative between steps and relies on the provider for the
  step itself - no mythical "instant hardware interrupt" is claimed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from jarvis.orchestration.turn import TurnCancellation
from jarvis.voice.errors import (
    VoiceCancelledError,
    VoiceError,
    clean_message,
    sanitize_text,
)
from jarvis.voice.models import VoiceState, VoiceTurn
from jarvis.voice.router import AUTO, VoiceRouter

logger = logging.getLogger(__name__)

VOICE_STATE_CHANGED = "voice.state_changed"

NO_SPEECH_MESSAGE = "речь не распознана: ход не отправлен"
DISABLED_MESSAGE = "голосовая подсистема отключена"


class VoiceSession:
    """Owns the voice states and drives one interaction at a time."""

    def __init__(
        self,
        router: VoiceRouter,
        *,
        dialog: Any | None = None,
        bus: Any | None = None,
        enabled: bool = True,
        stt_provider: str = AUTO,
        tts_provider: str = AUTO,
        language: str = "",
        listen_timeout_seconds: float = 8.0,
        speak_responses: bool = False,
        privacy: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._router = router
        self._dialog = dialog
        self._bus = bus
        self._enabled = bool(enabled)
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider
        self._language = (language or "").strip()
        self._listen_timeout = max(0.5, float(listen_timeout_seconds))
        self._speak_responses = bool(speak_responses)
        self._privacy = privacy
        self._clock = clock
        self._lock = threading.RLock()
        self._state = VoiceState.IDLE.value
        self._last_error: str | None = None
        self._cancellation: TurnCancellation | None = None
        self._last_transcript = ""

    # -- introspection ------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._state in {
                VoiceState.LISTENING.value,
                VoiceState.TRANSCRIBING.value,
                VoiceState.THINKING.value,
                VoiceState.SPEAKING.value,
                VoiceState.CANCELLING.value,
            }

    def snapshot(self) -> VoiceTurn:
        """Current state without starting anything."""
        with self._lock:
            return VoiceTurn(
                transcript=self._last_transcript,
                state=self._state,
                error=self._last_error or "",
            )

    def explain(self, *, probe: bool = True) -> dict[str, Any]:
        """Diagnostics: providers, availability, reasons. Never listens."""
        summary = dict(self._router.explain(probe=probe))
        summary.update(
            {
                "enabled": self._enabled,
                "state": self.state,
                "listen_timeout_seconds": self._listen_timeout,
                "speak_responses": self._speak_responses,
                "privacy": self._router.privacy,
                "last_error": self._last_error,
            }
        )
        if not self._enabled:
            summary["state"] = VoiceState.UNAVAILABLE.value
        return summary

    def describe(self) -> dict[str, Any]:
        """The same report as `explain()` - a stable name for the UI."""
        return self.explain(probe=False)

    # -- the lifecycle ------------------------------------------------------

    def transcribe(
        self,
        *,
        timeout_seconds: float | None = None,
        cancellation: TurnCancellation | None = None,
    ) -> VoiceTurn:
        """Listen once and return the recognised text. No dialogue turn.

        Used by the UI so a transcript enters the *same* path as typed text.
        An empty transcript is never treated as a request.
        """
        if not self._enabled:
            return self._finish(VoiceTurn(error=DISABLED_MESSAGE), VoiceState.UNAVAILABLE)

        cancellation = cancellation or TurnCancellation()
        with self._lock:
            self._cancellation = cancellation
            self._last_error = None

        timeout = self._listen_timeout if timeout_seconds is None else timeout_seconds

        try:
            selection = self._router.select_stt(self._stt_provider, probe=True)
        except VoiceError as exc:
            logger.info("stt selection failed: %s", clean_message(exc))
            return self._finish(
                VoiceTurn(error=clean_message(exc)), VoiceState.UNAVAILABLE
            )

        if cancellation.requested:
            return self._finish(VoiceTurn(), VoiceState.CANCELLED)

        provider = selection.provider
        self._publish_state(VoiceState.LISTENING, detail=selection.reason)
        try:
            transcription = provider.transcribe(
                timeout_seconds=timeout,
                language=self._language_hint(),
                cancellation=cancellation,
            )
        except VoiceCancelledError as exc:
            return self._finish(VoiceTurn(error=str(exc)), VoiceState.CANCELLED)
        except VoiceError as exc:
            return self._finish(VoiceTurn(error=clean_message(exc)), VoiceState.ERROR)

        if cancellation.requested:
            return self._finish(
                VoiceTurn(transcript=transcription.text), VoiceState.CANCELLED
            )

        self._publish_state(VoiceState.TRANSCRIBING, detail="")
        transcript = sanitize_text(transcription.text).strip()
        self._last_transcript = transcript
        if not transcript:
            logger.info("no speech recognised; nothing will be sent")
            return self._finish(
                VoiceTurn(transcript_empty=True, error=NO_SPEECH_MESSAGE), VoiceState.IDLE
            )
        # No terminal state yet: `listen()` continues into the dialogue turn,
        # and an intermediate "idle" would claim the work had finished.
        with self._lock:
            self._last_transcript = transcript
        return VoiceTurn(transcript=transcript, state=VoiceState.TRANSCRIBING.value)

    def listen(
        self, *, timeout_seconds: float | None = None, privacy: str | None = None
    ) -> VoiceTurn:
        """Listen once, transcribe, and - only then - run a dialogue turn."""
        cancellation = TurnCancellation()
        captured = self.transcribe(
            timeout_seconds=timeout_seconds, cancellation=cancellation
        )
        if captured.transcript_empty or not captured.transcript or captured.error:
            return captured
        return self._run_turn(captured.transcript, cancellation, privacy=privacy)

    def speak(self, text: str) -> bool:
        """Speak one answer. Empty text is ignored instead of playing silence."""
        message = (text or "").strip()
        if not message:
            logger.debug("speak called with empty text; nothing to play")
            return False
        if not self._enabled:
            self._publish_state(VoiceState.UNAVAILABLE, detail=DISABLED_MESSAGE)
            return False

        cancellation = TurnCancellation()
        with self._lock:
            self._cancellation = cancellation
        try:
            selection = self._router.select_tts(self._tts_provider, probe=True)
        except VoiceError as exc:
            self._last_error = clean_message(exc)
            self._publish_state(VoiceState.UNAVAILABLE, detail=self._last_error)
            logger.info("tts selection failed: %s", self._last_error)
            return False

        self._publish_state(VoiceState.SPEAKING, detail=selection.reason)
        try:
            selection.provider.speak(message, cancellation=cancellation)
        except VoiceCancelledError:
            self._publish_state(VoiceState.CANCELLED, detail="воспроизведение отменено")
            return False
        except VoiceError as exc:
            self._last_error = clean_message(exc)
            self._publish_state(VoiceState.ERROR, detail=self._last_error)
            return False
        self._publish_state(VoiceState.IDLE, detail="")
        return True

    def cancel(self) -> bool:
        """Ask the running operation to stop. Cooperative, not magical.

        Returns `False` when nothing is running: cancelling nothing is not a
        success.
        """
        with self._lock:
            cancellation = self._cancellation
            running = self.busy
        if cancellation is None or not running:
            return False
        cancellation.cancel()
        self._publish_state(VoiceState.CANCELLING, detail="запрошена отмена")
        return True

    def close(self) -> None:
        """Release the session: nothing is left running by ownership."""
        cancellation = self._cancellation
        if cancellation is not None and self.busy:
            cancellation.cancel()
            # A closed session must not keep claiming it is listening: the
            # operation is gone, so the state has to say so.
            self._publish_state(VoiceState.CANCELLED, detail="сессия закрыта")
        with self._lock:
            self._cancellation = None

    # -- internals ----------------------------------------------------------

    def _run_turn(
        self, transcript: str, cancellation: TurnCancellation, *, privacy: str | None
    ) -> VoiceTurn:
        """Send the transcript into the existing dialogue pipeline."""
        if self._dialog is None:
            return self._finish(
                VoiceTurn(transcript=transcript, error="диалог недоступен"),
                VoiceState.ERROR,
            )
        privacy_mode = privacy or self._privacy
        self._publish_state(VoiceState.THINKING, detail="ход отправлен в диалог")
        try:
            # Same call a typed message would use: same tools, same approvals.
            if privacy_mode:
                result = self._dialog.send(
                    transcript, cancellation=cancellation, privacy=privacy_mode
                )
            else:
                result = self._dialog.send(transcript, cancellation=cancellation)
        except Exception as exc:  # noqa: BLE001 - reported, never propagated raw
            return self._finish(
                VoiceTurn(transcript=transcript, error=clean_message(exc)),
                VoiceState.ERROR,
            )

        if getattr(result, "status", None) is not None and str(
            getattr(getattr(result, "status"), "value", result.status)
        ) == "cancelled":
            return self._finish(
                VoiceTurn(transcript=transcript), VoiceState.CANCELLED
            )

        pending = getattr(result, "pending", None)
        if pending is not None:
            # Voice has no special rights: the call waits like any other.
            return self._finish(
                VoiceTurn(
                    transcript=transcript,
                    pending=True,
                    pending_tool=str(getattr(pending, "tool", "")),
                    pending_id=str(getattr(pending, "pending_id", "")),
                ),
                VoiceState.IDLE,
            )

        answer = str(getattr(result, "answer", "") or "")
        failure = str(getattr(result, "error", "") or "")
        spoken = False
        if answer and self._speak_responses:
            spoken = self.speak(answer)
        if failure and not answer:
            self._last_error = sanitize_text(failure)
            return self._finish(
                VoiceTurn(transcript=transcript, error=self._last_error),
                VoiceState.ERROR,
            )
        return self._finish(
            VoiceTurn(transcript=transcript, answer=answer, spoken=spoken),
            VoiceState.IDLE,
        )

    def _language_hint(self) -> str:
        return self._language

    def _publish_state(self, state: VoiceState, *, detail: str) -> None:
        self._set_state(state.value, detail=detail)
        if self._bus is None:
            return
        try:
            from jarvis.core.bus import Event

            self._bus.publish(
                Event(
                    VOICE_STATE_CHANGED,
                    {"state": state.value, "detail": sanitize_text(detail)},
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics never break voice
            logger.debug("voice event not published: %s", exc)

    def _set_state(self, state: str, *, detail: str = "") -> None:
        with self._lock:
            previous = self._state
            self._state = state
            if detail and state in {
                VoiceState.ERROR.value,
                VoiceState.UNAVAILABLE.value,
            }:
                self._last_error = sanitize_text(detail)
        logger.debug("voice state %s -> %s", previous, state)

    def _finish(self, turn: VoiceTurn, state: VoiceState) -> VoiceTurn:
        with self._lock:
            self._cancellation = None
        finished = VoiceTurn(
            transcript=turn.transcript or self._last_transcript,
            answer=turn.answer,
            state=state.value,
            error=turn.error,
            transcript_empty=turn.transcript_empty,
            spoken=turn.spoken,
            pending=turn.pending,
            pending_tool=turn.pending_tool,
            pending_id=turn.pending_id,
        )
        self._publish_state(state, detail=turn.error)
        return finished
