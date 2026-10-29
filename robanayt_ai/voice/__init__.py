"""Voice subsystem (Phase 8).

What this layer is:

- A provider boundary for speech-to-text and text-to-speech, separate from AI
  model routing (Phase 7) - cheap models do not make ears cheaper.
- A session that turns words into an ordinary text request for the existing
  `DialogService`: the same tools, the same approvals, the same refusals.
- Honest availability: implemented, configured, available and tested on
  hardware are four different claims, kept apart everywhere.

What this layer is not:

- It never stores or logs audio, never sends audio off-machine unless an
  explicitly allowed remote path exists, and never lets a spoken utterance
  skip the permission system.
"""

from __future__ import annotations

from robanayt_ai.voice.errors import (
    SpeechError,
    TranscriptionError,
    VoiceCancelledError,
    VoiceConfigurationError,
    VoiceDisabledError,
    VoiceError,
    VoiceUnavailableError,
    clean_message,
)
from robanayt_ai.voice.models import (
    LOCAL,
    REMOTE,
    TERMINAL_VOICE_STATES,
    VOICE_STATE_LABELS,
    Transcription,
    VoiceCapabilities,
    VoiceProviderStatus,
    VoiceSelection,
    VoiceState,
    VoiceTurn,
)
from robanayt_ai.voice.provider import (
    SpeechToTextProvider,
    TextToSpeechProvider,
    VoiceProvider,
)
from robanayt_ai.voice.providers import SapiSpeechRecognizer, SapiSpeechSynthesizer
from robanayt_ai.voice.router import (
    ANY_PRIVACY,
    AUTO,
    LOCAL_ONLY,
    NONE,
    VALID_PRIVACY_MODES,
    VoiceRouter,
    normalize_privacy,
)
from robanayt_ai.voice.session import VOICE_STATE_CHANGED, VoiceSession

__all__ = [
    "ANY_PRIVACY",
    "AUTO",
    "LOCAL",
    "LOCAL_ONLY",
    "NONE",
    "REMOTE",
    "SapiSpeechRecognizer",
    "SapiSpeechSynthesizer",
    "SpeechError",
    "SpeechToTextProvider",
    "TERMINAL_VOICE_STATES",
    "Transcription",
    "TranscriptionError",
    "VOICE_STATE_CHANGED",
    "VOICE_STATE_LABELS",
    "VALID_PRIVACY_MODES",
    "VoiceCancelledError",
    "VoiceCapabilities",
    "VoiceConfigurationError",
    "VoiceDisabledError",
    "VoiceError",
    "VoiceProvider",
    "VoiceProviderStatus",
    "VoiceRouter",
    "VoiceSelection",
    "VoiceSession",
    "VoiceState",
    "VoiceTurn",
    "VoiceUnavailableError",
    "clean_message",
    "normalize_privacy",
    "TextToSpeechProvider",
]
