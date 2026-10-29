"""Voice data structures.

Everything here is metadata or text. No audio sample, byte of sound or raw
waveform is ever stored, logged or passed around: a transcript is ordinary
text and travels through the same dialogue pipeline as typed text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

# Where the audio is processed. This is what privacy decisions key off.
LOCAL = "local"
REMOTE = "remote"


class VoiceState(str, Enum):
    """States of the voice subsystem - real ones only."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


VOICE_STATE_LABELS: dict[str, str] = {
    VoiceState.IDLE.value: "микрофон неактивен",
    VoiceState.LISTENING.value: "слушаю",
    VoiceState.TRANSCRIBING.value: "распознаю речь",
    VoiceState.THINKING.value: "обработка запроса",
    VoiceState.SPEAKING.value: "говорю",
    VoiceState.CANCELLING.value: "отмена",
    VoiceState.CANCELLED.value: "отменено",
    VoiceState.ERROR.value: "ошибка",
    VoiceState.UNAVAILABLE.value: "голос недоступен",
}

# States that are already decided; the session leaves them by explicit action.
TERMINAL_VOICE_STATES: frozenset[str] = frozenset(
    {VoiceState.CANCELLED.value, VoiceState.ERROR.value, VoiceState.UNAVAILABLE.value}
)


@dataclass(frozen=True)
class VoiceCapabilities:
    """What a voice provider can really do. Nothing is assumed."""

    streaming: bool = False
    """True only when the backend really delivers partial results."""

    local: bool = True
    """True when no audio leaves the machine."""

    cancel_supported: bool = False
    """True when the backend can stop mid-operation, not only between steps."""


@dataclass(frozen=True)
class VoiceProviderStatus:
    """Availability report: what is configured, what was verified, and how."""

    provider: str
    kind: str = LOCAL
    configured: bool = False
    available: bool = False
    message: str = ""
    tested_on_hardware: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "kind": self.kind,
            "configured": self.configured,
            "available": self.available,
            "message": self.message,
            "tested_on_hardware": self.tested_on_hardware,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Transcription:
    """Recognised text. Empty text means "no speech understood"."""

    text: str
    provider: str = ""
    language: str = ""
    confidence: float | None = None
    partial: bool = False

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "text_length": len(self.text),
            "provider": self.provider,
            "language": self.language or None,
            "confidence": self.confidence,
            "partial": self.partial,
        }


@dataclass(frozen=True)
class VoiceTurn:
    """One voice interaction: what was heard, what the dialogue answered."""

    transcript: str = ""
    answer: str = ""
    state: str = VoiceState.IDLE.value
    error: str = ""
    transcript_empty: bool = False
    spoken: bool = False
    # A spoken request may also stop at an approval - exactly like text.
    pending: bool = False
    pending_tool: str = ""
    pending_id: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.transcript_empty

    def as_dict(self) -> dict[str, Any]:
        return {
            "transcript_length": len(self.transcript),
            "answer_length": len(self.answer),
            "state": self.state,
            "error": self.error or None,
            "transcript_empty": self.transcript_empty,
            "spoken": self.spoken,
            "pending": self.pending,
            "pending_tool": self.pending_tool or None,
        }


@dataclass(frozen=True)
class VoiceSelection:
    """Why a provider was picked - the same honesty as the AI router."""

    provider: Any = None
    reason: str = ""
    explicit: bool = False
    rejections: tuple[tuple[str, str], ...] = ()

    @property
    def name(self) -> str:
        return getattr(self.provider, "id", "") if self.provider is not None else ""
