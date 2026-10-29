"""Local Windows speech providers (SAPI / System.Speech).

These are **real** providers using capabilities that ship with Windows, so no
new Python dependency is needed. They still separate four facts, because they
are different on every machine:

- implemented: yes - this module is the implementation;
- configured: nothing to configure, so always true here;
- available: only when a probe really found a voice / a recognizer;
- tested_on_hardware: only after someone actually listened to it here.

Nothing here pretends availability. On a machine without installed voices or
without a recognition engine the providers report exactly that, with a hint
about what the user can do.
"""

from __future__ import annotations

import logging
from base64 import b64encode
from typing import Any, Callable, Mapping

from jarvis.voice.errors import (
    SpeechError,
    TranscriptionError,
    VoiceCancelledError,
    VoiceUnavailableError,
    clean_message,
)
from jarvis.voice.models import LOCAL, Transcription, VoiceCapabilities, VoiceProviderStatus
from jarvis.voice.providers.powershell import CommandResult, SubprocessPowerShellRunner

logger = logging.getLogger(__name__)

SYNTHESIZER_ID = "windows_sapi_tts"
RECOGNIZER_ID = "windows_sapi_stt"

PROBE_VOICES = """\
# kind=probe-voices - read-only: lists installed voices, speaks nothing.
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($voice in $s.GetInstalledVoices()) {
  'VOICE|' + $voice.VoiceInfo.Name + '|' + $voice.VoiceInfo.Culture
}
$s.Dispose()
"""

PROBE_RECOGNIZERS = """\
# kind=probe-recognizers - read-only: no microphone is opened here.
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
foreach ($engine in [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()) {
  'RECOGNIZER|' + $engine.Culture + '|' + $engine.Name
}
"""

SPEAK_SCRIPT_TEMPLATE = """\
# kind=speak - the only script that produces audio.
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$text = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{payload}'))
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
{select_voice}
$s.Speak($text)
$s.Dispose()
"""

SELECT_VOICE_TEMPLATE = """\
$name = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{voice_payload}'))
try {{ $s.SelectVoice($name) }} catch {{ }}
"""

RECOGNIZE_SCRIPT = """\
# kind=recognize - the only script that opens the microphone.
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$engines = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()
if ($engines.Count -eq 0) {{ Write-Output 'NO_RECOGNIZER'; exit 3 }}
$culture = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{culture}'))
if ($culture) {{
  $match = $engines | Where-Object {{ $_.Culture -like ($culture + '*') }} | Select-Object -First 1
  if ($null -eq $match) {{ Write-Output 'NO_CULTURE'; exit 4 }}
}}
$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$engine.SetInputToDefaultAudioDevice()
$engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
$result = $engine.Recognize([TimeSpan]::FromSeconds({seconds}))
if ($null -eq $result) {{ Write-Output 'NO_SPEECH'; exit 5 }}
Write-Output ('TEXT:' + $result.Text)
Write-Output ('CONFIDENCE:' + $result.Confidence)
"""


class SapiSpeechSynthesizer:
    """Text-to-speech through Windows System.Speech (local, no dependency)."""

    id = SYNTHESIZER_ID
    kind = LOCAL

    def __init__(
        self,
        *,
        voice: str = "",
        max_chars: int = 2000,
        timeout_seconds: float = 60.0,
        runner: Callable[..., CommandResult] | None = None,
    ) -> None:
        self._voice = (voice or "").strip()
        self._max_chars = max(1, int(max_chars))
        self._timeout = max(1.0, float(timeout_seconds))
        self._runner = runner or SubprocessPowerShellRunner()
        self._probed: bool = False
        self._voices: tuple[str, ...] = ()
        self._message: str = "доступность не проверена"

    @property
    def capabilities(self) -> VoiceCapabilities:
        # Playback stops when our own child process is terminated; a started
        # sentence may be cut mid-word, not rolled back.
        return VoiceCapabilities(local=True, streaming=False, cancel_supported=True)

    def status(self, *, probe: bool = False) -> VoiceProviderStatus:
        # A status without a probe answers honestly "not checked yet": probing
        # spawns a helper process, so it must never hide inside a UI frame.
        if probe:
            self._probe()
        return VoiceProviderStatus(
            provider=self.id,
            kind=self.kind,
            configured=True,
            available=bool(self._voices),
            message=self._message,
            metadata={"voices": list(self._voices), "voice_selected": self._voice or None},
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "provider": self.id,
            "kind": self.kind,
            "voice": self._voice or "по умолчанию",
            "max_chars": self._max_chars,
            "voices": list(self._voices),
        }

    def speak(self, text: str, *, cancellation: Any | None = None) -> None:
        """Speak `text`. Empty or oversized input is refused, not played."""
        message = (text or "").strip()
        if not message:
            return
        if len(message) > self._max_chars:
            raise SpeechError(
                f"текст длиннее лимита ({len(message)} > {self._max_chars} символов)"
            )
        # Verify right before doing something audible: availability may change
        # (a voice uninstalled, a policy applied) and silence would be worse
        # than a clear refusal.
        state = self.status(probe=True)
        if not state.available:
            raise VoiceUnavailableError(f"{self.id}: {state.message}")

        script = SPEAK_SCRIPT_TEMPLATE.format(
            payload=_b64(message),
            select_voice=(
                SELECT_VOICE_TEMPLATE.format(voice_payload=_b64(self._voice))
                if self._voice
                else ""
            ),
        )
        result = self._runner(
            script,
            timeout=self._timeout,
            cancel=(lambda: cancellation.requested) if cancellation is not None else None,
        )
        if result.cancelled or (cancellation is not None and cancellation.requested):
            raise VoiceCancelledError("воспроизведение отменено")
        if result.timed_out:
            raise SpeechError("синтез речи не завершился вовремя")
        if not result.ok:
            raise SpeechError(clean_message(result.error or "синтез речи завершился ошибкой"))
        logger.info("spoken %s characters via %s", len(message), self.id)

    def _probe(self) -> None:
        try:
            result = self._runner(PROBE_VOICES, timeout=15.0)
        except Exception as exc:  # noqa: BLE001 - probing must never crash
            self._probed = True
            self._voices = ()
            self._message = f"проба не удалась: {clean_message(exc)}"
            return
        voices = tuple(
            line.split("|", 2)[1]
            for line in result.lines()
            if line.startswith("VOICE|") and len(line.split("|", 2)) > 2
        )
        self._probed = True
        self._voices = voices
        if voices:
            self._message = f"найдено голосов: {len(voices)}"
        elif result.ok:
            self._message = "голоса Windows не установлены"
        else:
            self._message = clean_message(
                result.error or "PowerShell недоступен для System.Speech"
            )


class SapiSpeechRecognizer:
    """Speech-to-text through Windows System.Speech dictation grammar.

    The dictation engine reads the default input device itself, so no Python
    audio-capture library is involved. Real availability depends on installed
    recognition engines, which most Windows setups do not have - that is
    checked by a probe, not assumed.
    """

    id = RECOGNIZER_ID
    kind = LOCAL

    def __init__(
        self,
        *,
        default_language: str = "",
        runner: Callable[..., CommandResult] | None = None,
    ) -> None:
        self._default_language = (default_language or "").strip()
        self._runner = runner or SubprocessPowerShellRunner()
        self._probed = False
        self._recognizers: tuple[str, ...] = ()
        self._message = "доступность не проверена"

    @property
    def capabilities(self) -> VoiceCapabilities:
        # No partial results: dictation recognition returns one final phrase.
        return VoiceCapabilities(local=True, streaming=False, cancel_supported=True)

    def status(self, *, probe: bool = False) -> VoiceProviderStatus:
        if probe:
            self._probe()
        return VoiceProviderStatus(
            provider=self.id,
            kind=self.kind,
            configured=True,
            available=bool(self._recognizers),
            message=self._message,
            metadata={
                "recognizers": list(self._recognizers),
                "default_language": self._default_language or None,
            },
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "provider": self.id,
            "kind": self.kind,
            "default_language": self._default_language or "любой",
            "recognizers": list(self._recognizers),
        }

    def transcribe(
        self,
        *,
        timeout_seconds: float,
        language: str = "",
        cancellation: Any | None = None,
    ) -> Transcription:
        """Listen once and return recognised text. Empty text = no speech."""
        seconds = max(0.5, float(timeout_seconds))
        wanted = (language or self._default_language or "").strip()
        # A real check before touching the microphone, every time.
        state = self.status(probe=True)
        if not state.available:
            raise VoiceUnavailableError(f"{self.id}: {state.message}")
        if wanted and not self._supports(wanted):
            raise VoiceUnavailableError(
                f"{self.id}: нет распознавателя для языка {wanted!r} "
                f"(установлены: {', '.join(self._recognizers) or 'нет'})"
            )

        script = RECOGNIZE_SCRIPT.format(culture=_b64(wanted), seconds=seconds)
        result = self._runner(
            script,
            timeout=seconds + 15.0,
            cancel=(lambda: cancellation.requested) if cancellation is not None else None,
        )
        completed = any(line.startswith("TEXT:") for line in result.lines())
        if result.cancelled and not completed:
            raise VoiceCancelledError("распознавание отменено")
        if result.timed_out:
            raise TranscriptionError("распознавание не завершилось вовремя")

        text = ""
        confidence: float | None = None
        for line in result.lines():
            if line.startswith("TEXT:"):
                text = line[len("TEXT:") :].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[len("CONFIDENCE:") :].strip())
                except ValueError:
                    confidence = None
        if any(line.startswith("NO_RECOGNIZER") for line in result.lines()):
            raise VoiceUnavailableError(f"{self.id}: распознаватель не установлен")
        if any(line.startswith("NO_CULTURE") for line in result.lines()):
            raise VoiceUnavailableError(
                f"{self.id}: нет распознавателя для языка {wanted!r}"
            )
        if any(line.startswith("NO_SPEECH") for line in result.lines()):
            # Heard silence: that is a result, not a failure.
            logger.info("recognizer heard nothing within %.1fs", seconds)
            return Transcription(text="", provider=self.id, language=wanted or "")
        if not result.ok and not text:
            raise TranscriptionError(clean_message(result.error or "распознавание не удалось"))
        return Transcription(
            text=text,
            provider=self.id,
            language=wanted or (self._recognizers[0] if self._recognizers else ""),
            confidence=confidence,
        )

    def _supports(self, language: str) -> bool:
        wanted = language.lower()
        return any(item.lower().startswith(wanted) for item in self._recognizers)

    def _probe(self) -> None:
        try:
            result = self._runner(PROBE_RECOGNIZERS, timeout=15.0)
        except Exception as exc:  # noqa: BLE001 - probing must never crash
            self._probed = True
            self._recognizers = ()
            self._message = f"проба не удалась: {clean_message(exc)}"
            return
        cultures = tuple(
            line.split("|", 2)[1]
            for line in result.lines()
            if line.startswith("RECOGNIZER|") and len(line.split("|", 2)) > 2
        )
        self._probed = True
        self._recognizers = cultures
        if cultures:
            self._message = f"распознаватели: {', '.join(cultures)}"
        elif result.ok:
            self._message = (
                "движок распознавания есть, но ни один распознаватель речи "
                "не установлен (InstalledRecognizers=0)"
            )
        else:
            self._message = clean_message(
                result.error or "PowerShell недоступен для System.Speech"
            )


def _b64(text: str) -> str:
    """Base64 of UTF-8 - a payload format that cannot break quoting."""
    return b64encode(text.encode("utf-8")).decode("ascii")


__all__ = ["SapiSpeechRecognizer", "SapiSpeechSynthesizer"]
