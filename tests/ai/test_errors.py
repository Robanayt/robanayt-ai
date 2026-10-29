"""Error hierarchy and secret sanitisation."""

from __future__ import annotations

import pytest

from jarvis.ai.errors import (
    AIError,
    AIRequestError,
    AIDisabledError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    REDACTED,
    redact,
    sanitize,
    sanitize_headers,
)

ALL_ERRORS = (
    AIRequestError,
    AIDisabledError,
    ProviderError,
    ProviderUnavailableError,
    ProviderConfigurationError,
    ProviderTimeoutError,
    ProviderResponseError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
)


def test_every_error_derives_from_ai_error() -> None:
    for error_type in ALL_ERRORS:
        assert issubclass(error_type, AIError)


def test_provider_error_carries_provider_and_status() -> None:
    error = ProviderAuthenticationError("denied", provider="openai_compatible", status=401)

    assert error.provider == "openai_compatible"
    assert error.status == 401
    assert "[openai_compatible]" in str(error)
    assert "denied" in str(error)


def test_provider_error_without_provider_is_plain() -> None:
    assert str(ProviderError("boom")) == "boom"


def test_redact_marks_any_value() -> None:
    assert redact("secret-value") == REDACTED
    assert redact(None) == ""
    assert redact("") == ""


def test_sanitize_removes_known_secrets() -> None:
    secret = "sk-live-abcdef123456"
    text = f"request failed with key {secret} and status 401"

    result = sanitize(text, [secret])

    assert secret not in result
    assert REDACTED in result
    assert "status 401" in result


def test_sanitize_ignores_short_values() -> None:
    # Very short values are not treated as secrets (avoid over-redaction).
    assert sanitize("value ab appeared", ["ab"]) == "value ab appeared"


def test_sanitize_accepts_non_string_input() -> None:
    assert sanitize(KeyError("boom"), []) == "'boom'"


def test_sanitize_headers_masks_credentials() -> None:
    headers = {
        "Authorization": "Bearer sk-live-abcdef123456",
        "OpenAI-Organization": "org-42",
        "Content-Type": "application/json",
    }

    safe = sanitize_headers(headers)

    assert safe["Authorization"] == REDACTED
    assert safe["OpenAI-Organization"] == REDACTED
    assert safe["Content-Type"] == "application/json"
    assert "sk-live-abcdef123456" not in str(safe)


def test_sanitize_headers_handles_empty_input() -> None:
    assert sanitize_headers(None) == {}
    assert sanitize_headers({}) == {}


def test_error_messages_never_carry_headers() -> None:
    secret = "sk-live-abcdef123456"
    message = sanitize(f"HTTP 401: unauthorized for key {secret}", [secret])

    with pytest.raises(ProviderAuthenticationError) as info:
        raise ProviderAuthenticationError(message, provider="openai_compatible")

    assert secret not in str(info.value)
