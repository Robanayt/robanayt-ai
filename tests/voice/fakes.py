"""Test doubles for the voice layer.

These fakes exist for unit tests only. They are never registered as usable
providers in the application, and nothing here is claimed to be real audio:
every provider states in its metadata that it was not tested on hardware.
"""

from __future__ import annotations

from typing import Any, Callable

from robanayt_ai.voice.errors import (
    SpeechError,
    TranscriptionError,
    VoiceCancelledError,
    VoiceUnavailableError,
)
from robanayt_ai.voice.models import (
    LOCAL,
    REMOTE,
    Transcription,
    VoiceCapabilities,
    VoiceProviderStatus,
)
from robanayt_ai.voice.providers.powershell import CommandResult


class ScriptedSTT:
    """Speech-to-text double: returns scripted text, errors or cancellation."""

    def __init__(
        self,
        transcript: str = "привет джарвис",
        *,
        provider_id: str = "scripted_stt",
        kind: str = LOCAL,
        error: Exception | None = None,
        available: bool = True,
        configured: bool = True,
        message: str = "готов",
        cancel_requested: bool = False,
        capabilities: VoiceCapabilities | None = None,
    ) -> None:
        self.id = provider_id
        self.kind = kind
        self.capabilities = capabilities or VoiceCapabilities(
            streaming=False, local=kind == LOCAL, cancel_supported=True
        )
        self.transcript = transcript
        self.error = error
        self._available = available
        self._configured = configured
        self._message = message
        self._cancel_requested = cancel_requested
        self.calls: list[dict[str, Any]] = []

    def status(self, *, probe: bool = False) -> VoiceProviderStatus:
        return VoiceProviderStatus(
            provider=self.id,
            kind=self.kind,
            configured=self._configured,
            available=self._available,
            message=self._message,
            metadata={"scripted": True},
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.id, "kind": self.kind, "scripted": True}

    def transcribe(
        self,
        *,
        timeout_seconds: float,
        language: str = "",
        cancellation: Any | None = None,
    ) -> Transcription:
        self.calls.append(
            {"timeout": timeout_seconds, "language": language, "cancel": cancellation}
        )
        if cancellation is not None and cancellation.requested:
            raise VoiceCancelledError("распознавание отменено")
        if self.error is not None:
            raise self.error
        if not self._available:
            raise VoiceUnavailableError(f"{self.id}: {self._message}")
        return Transcription(
            text=self.transcript, provider=self.id, language=language or "ru-RU"
        )


class UnavailableSTT(ScriptedSTT):
    """A provider that exists as a class but cannot listen - by design."""

    def __init__(self, message: str = "микрофон не найден") -> None:
        super().__init__(
            "",
            provider_id="unavailable_stt",
            available=False,
            configured=True,
            message=message,
        )


class RemoteSTT(ScriptedSTT):
    """Cloud speech-to-text double - used to prove privacy boundaries."""

    def __init__(self, transcript: str = "распознано в облаке") -> None:
        super().__init__(
            transcript, provider_id="remote_stt", kind=REMOTE, message="облачный провайдер"
        )


class ScriptedTTS:
    """Text-to-speech double: records everything it was asked to say."""

    def __init__(
        self,
        *,
        provider_id: str = "scripted_tts",
        kind: str = LOCAL,
        error: Exception | None = None,
        available: bool = True,
        configured: bool = True,
        message: str = "готов",
    ) -> None:
        self.id = provider_id
        self.kind = kind
        self.capabilities = VoiceCapabilities(
            streaming=False, local=kind == LOCAL, cancel_supported=True
        )
        self.error = error
        self._available = available
        self._configured = configured
        self._message = message
        self.spoken: list[str] = []

    def status(self, *, probe: bool = False) -> VoiceProviderStatus:
        return VoiceProviderStatus(
            provider=self.id,
            kind=self.kind,
            configured=self._configured,
            available=self._available,
            message=self._message,
            metadata={"scripted": True},
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.id, "kind": self.kind, "scripted": True}

    def speak(self, text: str, *, cancellation: Any | None = None) -> None:
        if not text or not text.strip():
            return
        self.spoken.append(text)
        if cancellation is not None and cancellation.requested:
            raise VoiceCancelledError("воспроизведение отменено")
        if self.error is not None:
            raise self.error
        if not self._available:
            raise VoiceUnavailableError(f"{self.id}: {self._message}")


class RecordingDialog:
    """Stand-in dialogue service: records nothing except what it received."""

    def __init__(self, answer: str = "ответ ассистента", *, error: str = "") -> None:
        self.answer_text = answer
        self.error_text = error
        self.calls: list[dict[str, Any]] = []

    def send(self, message: str, **kwargs: Any) -> Any:
        self.calls.append({"message": message, **kwargs})
        return _Result(
            answer=self.answer_text, error=self.error_text, status_value="success"
        )


class _Result:
    def __init__(self, *, answer: str, error: str, status_value: str) -> None:
        self.answer = answer
        self.error = error
        self.status = _Status(status_value)
        self.tool_calls: tuple[Any, ...] = ()
        self.conversation_id = None


class _Status:
    def __init__(self, value: str) -> None:
        self.value = value


class ScriptedRunner:
    """PowerShell stand-in: answers probes and scripts from a table.

    It never spawns a process, and it records every script it was given so a
    test can prove what exactly would have been asked of Windows.
    """

    def __init__(self, outputs: dict[str, CommandResult] | None = None) -> None:
        self.outputs = outputs or {}
        self.scripts: list[str] = []
        self.stdin: list[str] = []

    def __call__(
        self,
        script: str,
        *,
        timeout: float,
        stdin: str = "",
        cancel: Callable[[], bool] | None = None,
    ) -> CommandResult:
        self.scripts.append(script)
        self.stdin.append(stdin)
        if cancel is not None and cancel():
            return CommandResult(ok=False, code=-1, error="отменено", cancelled=True)
        # Longest marker wins: a recognise script also mentions recognizers,
        # so a short probe marker would otherwise shadow the intended answer.
        for marker in sorted(self.outputs, key=len, reverse=True):
            if marker in script:
                return self.outputs[marker]
        return CommandResult(ok=True, code=0, output="")


def command_ok(output: str = "") -> CommandResult:
    return CommandResult(ok=True, code=0, output=output)


def command_fail(output: str = "", error: str = "сбой PowerShell") -> CommandResult:
    return CommandResult(ok=False, code=1, output=output, error=error)


__all__ = [
    "REMOTE",
    "RemoteSTT",
    "RecordingDialog",
    "ScriptedRunner",
    "ScriptedSTT",
    "ScriptedTTS",
    "SpeechError",
    "TranscriptionError",
    "UnavailableSTT",
    "command_fail",
    "command_ok",
]
