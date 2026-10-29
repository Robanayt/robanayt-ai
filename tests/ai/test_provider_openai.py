"""OpenAI-compatible provider: parsing, auth, secrets, streaming."""

from __future__ import annotations

import pytest
from tests.ai.fakes import FakeTransport, openai_chat_response, sse_lines

from jarvis.ai.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from jarvis.ai.http import KIND_AUTH, KIND_CONNECTION, KIND_PARSE, KIND_RATE_LIMIT, KIND_TIMEOUT, TransportError
from jarvis.ai.models import AIRequest, AIMessage
from jarvis.ai.providers.openai_compatible import OpenAICompatibleProvider

# Obviously fake placeholder - never a real credential.
FAKE_KEY = "FAKE-TEST-KEY-0000"


def _provider(**kwargs) -> OpenAICompatibleProvider:
    defaults = {
        "base_url": "https://api.example.test/v1",
        "model": "test-model",
        "allow_remote": True,
        "env": {"JARVIS_OPENAI_API_KEY": FAKE_KEY},
    }
    defaults.update(kwargs)
    return OpenAICompatibleProvider(**defaults)


def _request() -> AIRequest:
    return AIRequest(messages=[AIMessage.user("hi")])


def test_status_without_base_url() -> None:
    status = OpenAICompatibleProvider(model="m").status()

    assert status.configured is False
    assert "base_url" in status.message


def test_status_without_model() -> None:
    status = OpenAICompatibleProvider(base_url="https://api.example.test/v1").status()

    assert status.configured is False
    assert "model" in status.message


def test_status_without_api_key_names_the_variable() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.example.test/v1", model="m", env={}
    )

    status = provider.status()

    assert status.configured is True
    assert status.available is False
    assert "JARVIS_OPENAI_API_KEY" in status.message
    assert FAKE_KEY not in status.message


def test_remote_disabled_by_default() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.example.test/v1",
        model="m",
        env={"JARVIS_OPENAI_API_KEY": FAKE_KEY},
    )

    status = provider.status()

    assert status.available is False
    assert "allow_remote" in status.message


def test_localhost_is_allowed_without_remote_flag() -> None:
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        env={"JARVIS_OPENAI_API_KEY": FAKE_KEY},
    )

    assert provider.status().available is True


def test_status_without_probe_does_not_call_transport() -> None:
    transport = FakeTransport()
    provider = _provider(transport=transport)

    provider.status()

    assert transport.calls == []


def test_probe_success_marks_available() -> None:
    transport = FakeTransport(get_result={"data": []})
    provider = _provider(transport=transport)

    status = provider.status(probe=True)

    assert status.available is True
    assert transport.calls[0]["url"].endswith("/models")


def test_probe_failure_is_reported_not_raised() -> None:
    transport = FakeTransport(error=TransportError(KIND_CONNECTION, "refused"))
    provider = _provider(transport=transport)

    status = provider.status(probe=True)

    assert status.available is False
    assert "probe failed" in status.message


def test_generate_parses_choices_and_usage() -> None:
    transport = FakeTransport(post_result=openai_chat_response("hello"))
    provider = _provider(transport=transport)

    response = provider.generate(_request())

    assert response.content == "hello"
    assert response.provider == "openai_compatible"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 20
    call = transport.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"


def test_organization_header_is_sent_when_configured() -> None:
    transport = FakeTransport(post_result=openai_chat_response("hi"))
    provider = _provider(organization="org-42", transport=transport)

    provider.generate(_request())

    assert transport.calls[0]["headers"]["OpenAI-Organization"] == "org-42"


def test_missing_api_key_is_configuration_error() -> None:
    provider = _provider(env={}, transport=FakeTransport())

    with pytest.raises(ProviderConfigurationError, match="API key missing"):
        provider.generate(_request())


def test_authentication_failure_is_typed_and_sanitized() -> None:
    error = TransportError(KIND_AUTH, f"HTTP 401: bad key {FAKE_KEY}", status=401)
    provider = _provider(transport=FakeTransport(error=error))

    with pytest.raises(ProviderAuthenticationError) as info:
        provider.generate(_request())

    assert FAKE_KEY not in str(info.value)
    assert info.value.status == 401


def test_rate_limit_is_typed() -> None:
    provider = _provider(
        transport=FakeTransport(error=TransportError(KIND_RATE_LIMIT, "HTTP 429", status=429))
    )

    with pytest.raises(ProviderRateLimitError) as info:
        provider.generate(_request())

    assert info.value.status == 429


def test_timeout_and_connection_mapping() -> None:
    with pytest.raises(ProviderTimeoutError):
        _provider(
            transport=FakeTransport(error=TransportError(KIND_TIMEOUT, "slow"))
        ).generate(_request())

    with pytest.raises(ProviderUnavailableError):
        _provider(
            transport=FakeTransport(error=TransportError(KIND_CONNECTION, "refused"))
        ).generate(_request())


def test_malformed_payload_is_response_error() -> None:
    provider = _provider(transport=FakeTransport(post_result={"choices": []}))

    with pytest.raises(ProviderResponseError, match="no choices"):
        provider.generate(_request())


def test_error_object_in_payload_is_response_error() -> None:
    provider = _provider(
        transport=FakeTransport(post_result={"error": {"message": "nope"}})
    )

    with pytest.raises(ProviderResponseError, match="nope"):
        provider.generate(_request())


def test_parse_error_is_response_error() -> None:
    provider = _provider(transport=FakeTransport(error=TransportError(KIND_PARSE, "bad")))

    with pytest.raises(ProviderResponseError):
        provider.generate(_request())


def test_stream_parses_sse() -> None:
    provider = _provider(transport=FakeTransport(stream_lines=sse_lines(["Hel", "lo"])))

    chunks = list(provider.stream(_request()))

    assert "".join(c.delta for c in chunks) == "Hello"
    assert chunks[-1].done is True


def test_stream_handles_done_sentinel() -> None:
    provider = _provider(transport=FakeTransport(stream_lines=["data: [DONE]"]))

    chunks = list(provider.stream(_request()))

    assert chunks[-1].done is True
    assert chunks[-1].delta == ""


def test_repr_and_describe_never_expose_the_key() -> None:
    provider = _provider()

    assert FAKE_KEY not in repr(provider)
    summary = provider.describe()
    assert summary["api_key_present"] is True
    assert FAKE_KEY not in str(summary)
    assert summary["api_key_env"] == "JARVIS_OPENAI_API_KEY"
