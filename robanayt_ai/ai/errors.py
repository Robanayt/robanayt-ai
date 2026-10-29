"""AI layer errors and secret-sanitisation helpers.

Rules enforced here:

- Provider failures are explicit typed errors, never bare `RuntimeError`.
- Nothing that could carry a credential is ever put into an exception
  message, a log record or a status dictionary: headers and secrets are
  redacted before they can escape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "***"

_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "openai-organization",
        "cookie",
        "set-cookie",
    }
)


class AIError(Exception):
    """Base class for every AI-layer failure."""


class AIRequestError(AIError):
    """The request itself is invalid (no messages, bad parameters, ...)."""


class AIDisabledError(AIError):
    """The AI layer is disabled by configuration."""


class ProviderError(AIError):
    """Base class for provider failures; carries the provider name."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status: int | None = None,
    ) -> None:
        self.provider = provider
        self.status = status
        self.raw_message = message
        super().__init__(f"[{provider}] {message}" if provider else message)


class ProviderUnavailableError(ProviderError):
    """Provider cannot be reached (connection refused, DNS, offline)."""


class ProviderConfigurationError(ProviderError):
    """Provider is misconfigured (missing model, base URL or API key)."""


class ProviderTimeoutError(ProviderError):
    """Provider did not answer within the configured timeout."""


class ProviderResponseError(ProviderError):
    """Provider answered, but the payload was malformed or unusable."""


class ProviderAuthenticationError(ProviderError):
    """Provider rejected the credentials (HTTP 401/403)."""


class ProviderRateLimitError(ProviderError):
    """Provider is rate limiting (HTTP 429)."""


def redact(value: str | None) -> str:
    """Return a non-reversible placeholder for a secret value."""
    if value is None:
        return ""
    return REDACTED if value else ""


def sanitize(text: Any, secrets: Iterable[str | None] = ()) -> str:
    """Remove known secret values from arbitrary text (errors, logs)."""
    result = text if isinstance(text, str) else str(text)
    for secret in secrets:
        if secret and len(secret) >= 4 and secret in result:
            result = result.replace(secret, REDACTED)
    return result


def sanitize_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Return headers safe for logging: sensitive values replaced."""
    if not headers:
        return {}
    safe: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _SENSITIVE_HEADERS:
            safe[key] = REDACTED
        else:
            safe[key] = value
    return safe
