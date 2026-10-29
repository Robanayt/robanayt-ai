"""Ollama provider: parsing, failures, streaming - all with a fake transport."""

from __future__ import annotations

import json

import pytest
from tests.ai.fakes import FakeTransport, ollama_chat_response

from robanayt_ai.ai.errors import (
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from robanayt_ai.ai.http import (
    KIND_CONNECTION,
    KIND_PARSE,
    KIND_TIMEOUT,
    TransportError,
)
from robanayt_ai.ai.models import AIRequest, AIMessage
from robanayt_ai.ai.providers.ollama import DEFAULT_BASE_URL, OllamaProvider


def _request(model: str | None = None) -> AIRequest:
    return AIRequest(messages=[AIMessage.user("hi")], model=model)


def test_defaults_use_localhost_and_no_model() -> None:
    provider = OllamaProvider()

    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.model == ""
    assert provider.capabilities.streaming is True
    assert provider.kind == "local"


def test_construction_does_not_touch_the_network() -> None:
    transport = FakeTransport(error=TransportError(KIND_CONNECTION, "should not happen"))

    provider = OllamaProvider(model="m", transport=transport)

    assert provider.model == "m"
    assert transport.calls == []


def test_status_without_model_reports_not_configured() -> None:
    transport = FakeTransport()
    provider = OllamaProvider(transport=transport)

    status = provider.status()

    assert status.configured is False
    assert status.available is False
    assert "no model configured" in status.message
    assert transport.calls == []


def test_status_without_probe_never_calls_transport() -> None:
    transport = FakeTransport(get_result={"models": []})
    provider = OllamaProvider(model="m", transport=transport)

    status = provider.status()

    assert status.configured is True
    assert status.available is True
    assert transport.calls == []


def test_probe_detects_missing_model() -> None:
    transport = FakeTransport(get_result={"models": [{"name": "other:latest"}]})
    provider = OllamaProvider(model="m", transport=transport)

    status = provider.status(probe=True)

    assert status.configured is True
    assert status.available is False
    assert "not found" in status.message
    assert transport.calls[0]["url"].endswith("/api/tags")


def test_probe_detects_present_model() -> None:
    transport = FakeTransport(get_result={"models": [{"name": "m"}]})
    provider = OllamaProvider(model="m", transport=transport)

    status = provider.status(probe=True)

    assert status.available is True
    assert "present locally" in status.message


def test_probe_reports_connection_failure_without_raising() -> None:
    transport = FakeTransport(error=TransportError(KIND_CONNECTION, "refused"))
    provider = OllamaProvider(model="m", transport=transport)

    status = provider.status(probe=True)

    assert status.available is False
    assert "probe failed" in status.message


def test_generate_parses_chat_response() -> None:
    transport = FakeTransport(post_result=ollama_chat_response("hello", model="m"))
    provider = OllamaProvider(model="m", transport=transport)

    response = provider.generate(_request())

    assert response.content == "hello"
    assert response.provider == "ollama"
    assert response.model == "m"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 18
    call = transport.calls[0]
    assert call["url"].endswith("/api/chat")
    assert call["payload"]["stream"] is False
    assert call["payload"]["messages"][0] == {"role": "user", "content": "hi"}


def test_generate_includes_options_when_set() -> None:
    transport = FakeTransport(post_result=ollama_chat_response("ok"))
    provider = OllamaProvider(model="m", transport=transport)

    provider.generate(AIRequest(messages=[AIMessage.user("hi")], temperature=0.4, max_tokens=32))

    options = transport.calls[0]["payload"]["options"]
    assert options["temperature"] == 0.4
    assert options["num_predict"] == 32


def test_generate_without_model_is_configuration_error() -> None:
    provider = OllamaProvider(transport=FakeTransport())

    with pytest.raises(ProviderConfigurationError, match="no model configured"):
        provider.generate(_request())


def test_connection_failure_maps_to_unavailable() -> None:
    provider = OllamaProvider(
        model="m", transport=FakeTransport(error=TransportError(KIND_CONNECTION, "refused"))
    )

    with pytest.raises(ProviderUnavailableError):
        provider.generate(_request())


def test_timeout_maps_to_timeout_error() -> None:
    provider = OllamaProvider(
        model="m", transport=FakeTransport(error=TransportError(KIND_TIMEOUT, "too slow"))
    )

    with pytest.raises(ProviderTimeoutError):
        provider.generate(_request())


def test_malformed_response_is_response_error() -> None:
    provider = OllamaProvider(model="m", transport=FakeTransport(post_result={"nope": 1}))

    with pytest.raises(ProviderResponseError, match="no text content"):
        provider.generate(_request())


def test_non_object_response_is_response_error() -> None:
    provider = OllamaProvider(model="m", transport=FakeTransport(post_result=["weird"]))

    with pytest.raises(ProviderResponseError, match="unexpected response type"):
        provider.generate(_request())


def test_parse_error_maps_to_response_error() -> None:
    provider = OllamaProvider(
        model="m", transport=FakeTransport(error=TransportError(KIND_PARSE, "bad json"))
    )

    with pytest.raises(ProviderResponseError):
        provider.generate(_request())


def test_stream_parses_ndjson_lines() -> None:
    lines = [
        json.dumps({"message": {"content": "He"}, "done": False}),
        json.dumps({"message": {"content": "llo"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}),
    ]
    provider = OllamaProvider(model="m", transport=FakeTransport(stream_lines=lines))

    chunks = list(provider.stream(_request()))

    assert "".join(c.delta for c in chunks) == "Hello"
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"
    assert provider._transport.calls[0]["method"] == "STREAM"


def test_stream_rejects_malformed_line() -> None:
    provider = OllamaProvider(model="m", transport=FakeTransport(stream_lines=["{oops"]))

    with pytest.raises(ProviderResponseError, match="malformed stream line"):
        list(provider.stream(_request()))


def test_stream_reports_provider_error_field() -> None:
    provider = OllamaProvider(
        model="m", transport=FakeTransport(stream_lines=[json.dumps({"error": "boom"})])
    )

    with pytest.raises(ProviderResponseError, match="boom"):
        list(provider.stream(_request()))


def test_describe_is_secret_free() -> None:
    provider = OllamaProvider(model="m")

    summary = provider.describe()

    assert summary["provider"] == "ollama"
    assert summary["base_url"] == DEFAULT_BASE_URL
    assert "key" not in summary
    assert "m" in repr(provider)
