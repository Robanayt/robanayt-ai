"""Voice provider contracts.

A provider is a boundary, not a promise: a class existing says nothing about
availability. Every provider therefore separates four different things -

- implemented: the code path exists;
- configured: the provider has what it needs (a voice name, a URL, a device);
- available: a probe confirms it can actually work here and now;
- tested_on_hardware: someone really exercised it in this environment.

Only the last two may justify telling the user that voice works.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from robanayt_ai.voice.models import Transcription, VoiceCapabilities, VoiceProviderStatus


@runtime_checkable
class VoiceProvider(Protocol):
    """Common identity and introspection of every voice provider."""

    @property
    def id(self) -> str:
        """Stable identifier used in configuration and diagnostics."""

    @property
    def kind(self) -> str:
        """`local` or `remote` - decides whether audio may leave the machine."""

    @property
    def capabilities(self) -> VoiceCapabilities: ...

    def status(self, *, probe: bool = False) -> VoiceProviderStatus:
        """Configuration/availability report. A probe may touch the system."""

    def describe(self) -> Mapping[str, Any]:
        """Diagnostics with no secrets, no audio and no transcripts."""


@runtime_checkable
class SpeechToTextProvider(VoiceProvider, Protocol):
    """Microphone (or audio source) → text."""

    def transcribe(
        self,
        *,
        timeout_seconds: float,
        language: str = "",
        cancellation: Any | None = None,
    ) -> Transcription:
        """Listen once and return recognised text.

        Implementations check `cancellation` between steps; a backend that
        cannot be interrupted reports that through its capabilities rather
        than pretending it stopped immediately.
        """


@runtime_checkable
class TextToSpeechProvider(VoiceProvider, Protocol):
    """Text → audio output."""

    def speak(self, text: str, *, cancellation: Any | None = None) -> None:
        """Speak `text`. Empty text is ignored, never played as silence."""
