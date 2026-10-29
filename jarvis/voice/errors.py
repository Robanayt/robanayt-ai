"""Voice layer errors and message cleaning.

Rules enforced here:

- Provider failures are explicit typed errors, never bare `RuntimeError`.
- No audio payload ever enters an exception, log or status dictionary: only
  text length and language may be reported, never the samples themselves.
- Secrets (a remote API key, a token) are redacted before anything escapes -
  this layer never logs a credential, even when a provider has one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

REDACTED = "***"

# How much text may be quoted back in an error. Long enough to be useful,
# short enough that a whole paragraph never lands in a log line.
MAX_MESSAGE_CHARS = 300


class VoiceError(Exception):
    """Base class for every voice-layer failure."""


class VoiceDisabledError(VoiceError):
    """The voice subsystem is disabled by configuration."""


class VoiceConfigurationError(VoiceError):
    """The subsystem or a provider is misconfigured (bad name, empty id)."""


class VoiceUnavailableError(VoiceError):
    """Nothing usable was found: no microphone, no recognizer, no voice."""


class VoiceCancelledError(VoiceError):
    """The operation was stopped cooperatively at a safe point."""


class TranscriptionError(VoiceError):
    """Speech-to-text failed: device busy, no speech, backend failure."""


class SpeechError(VoiceError):
    """Text-to-speech failed: no voice, backend failure, refused input."""


def redact(value: str | None) -> str:
    """Return a non-reversible placeholder for a secret value."""
    if value is None:
        return ""
    return REDACTED if value else ""


def sanitize_text(text: object, secrets: Iterable[str | None] = ()) -> str:
    """Remove secrets and control characters from arbitrary text."""
    result = text if isinstance(text, str) else str(text)
    for secret in secrets:
        if secret and len(secret) >= 4 and secret in result:
            result = result.replace(secret, REDACTED)
    # Keep every reported line single-line: a provider must not inject
    # newlines into logs or UI labels.
    cleaned = "".join(
        character if character.isprintable() else " " for character in result
    ).strip()
    return cleaned


def clean_message(exc: BaseException, secrets: Iterable[str | None] = ()) -> str:
    """One-line, sanitised and length-bounded report of a failure."""
    text = sanitize_text(exc, secrets)
    if len(text) > MAX_MESSAGE_CHARS:
        return text[: MAX_MESSAGE_CHARS - 3] + "..."
    return text


def clean_mapping(data: Mapping[str, object]) -> dict[str, object]:
    """Copy a metadata mapping with every value sanitised for display."""
    return {str(key): sanitize_text(value) for key, value in data.items()}
