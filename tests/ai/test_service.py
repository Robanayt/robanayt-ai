"""AIService orchestration: validation, routing, retries, streaming."""

from __future__ import annotations

import pytest
from tests.ai.fakes import FakeProvider

from robanayt_ai.ai.context import ConversationContext
from robanayt_ai.ai.errors import (
    AIDisabledError,
    AIRequestError,
    ProviderAuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from robanayt_ai.ai.models import AIRequest, AIMessage
from robanayt_ai.ai.router import ProviderRouter, RoutingPolicy
from robanayt_ai.ai.service import AIService
from robanayt_ai.ai.streaming import collect_text


def _service(provider: FakeProvider, **kwargs) -> AIService:
    router = ProviderRouter([provider], policy=RoutingPolicy.LOCAL_FIRST)
    return AIService(router, **kwargs)


def test_generate_returns_normalised_response() -> None:
    provider = FakeProvider("ollama", reply="hello there")
    service = _service(provider, default_model="m")

    response = service.generate(prompt="hi")

    assert response.content == "hello there"
    assert response.provider == "ollama"
    assert response.model == "m"


def test_generate_accepts_existing_request() -> None:
    provider = FakeProvider("ollama")
    service = _service(provider)
    request = AIRequest(messages=[AIMessage.user("hi")], model="given")

    response = service.generate(request)

    assert provider.calls[0].model == "given"
    assert provider.calls[0].messages == request.messages
    assert response.model == "given"


def test_default_model_is_applied_when_request_has_none() -> None:
    provider = FakeProvider("ollama")
    service = _service(provider, default_model="fallback-model")

    service.generate(prompt="hi")

    assert provider.calls[0].model == "fallback-model"


def test_disabled_service_refuses_generation() -> None:
    service = _service(FakeProvider("ollama"), enabled=False)

    with pytest.raises(AIDisabledError):
        service.generate(prompt="hi")


def test_invalid_request_is_rejected() -> None:
    service = _service(FakeProvider("ollama"))

    with pytest.raises(AIRequestError):
        service.generate(AIRequest(messages=()))


def test_build_request_from_prompt_and_system() -> None:
    service = _service(FakeProvider("ollama"))

    request = service.build_request(prompt="hi", system="be brief")

    assert [m.role for m in request.messages] == ["system", "user"]


def test_build_request_from_context() -> None:
    context = ConversationContext("ctx system")
    context.add_user("earlier")
    service = _service(FakeProvider("ollama"))

    request = service.build_request(context=context, prompt="now")

    assert [m.role for m in request.messages] == ["system", "user", "user"]


def test_build_request_rejects_conflicting_sources() -> None:
    service = _service(FakeProvider("ollama"))
    context = ConversationContext()

    with pytest.raises(AIRequestError):
        service.build_request(context=context, messages=[AIMessage.user("x")])


def test_retry_on_timeout_then_success() -> None:
    delays: list[float] = []
    provider = FakeProvider(
        "ollama",
        failures=[ProviderTimeoutError("slow", provider="ollama")],
        reply="recovered",
    )
    service = _service(
        provider,
        max_retries=1,
        retry_delay_seconds=0.01,
        sleeper=delays.append,
    )

    response = service.generate(prompt="hi")

    assert response.content == "recovered"
    assert len(provider.calls) == 2
    assert delays == [0.01]


def test_no_retry_on_authentication_error() -> None:
    provider = FakeProvider(
        "ollama", failures=[ProviderAuthenticationError("denied", provider="ollama")]
    )
    service = _service(provider, max_retries=3)

    with pytest.raises(ProviderAuthenticationError):
        service.generate(prompt="hi")

    assert len(provider.calls) == 1


def test_retry_exhausted_raises_last_error() -> None:
    provider = FakeProvider(
        "ollama",
        failures=[
            ProviderUnavailableError("down", provider="ollama"),
            ProviderUnavailableError("still down", provider="ollama"),
        ],
    )
    service = _service(provider, max_retries=1)

    with pytest.raises(ProviderUnavailableError):
        service.generate(prompt="hi")

    assert len(provider.calls) == 2


def test_stream_returns_chunks() -> None:
    provider = FakeProvider("ollama", reply="streamed words here")
    service = _service(provider)

    text = collect_text(service.stream(prompt="hi"))

    assert text.strip() == "streamed words here"


def test_stream_falls_back_for_non_streaming_provider() -> None:
    provider = FakeProvider("ollama", reply="single shot", streaming=False)
    service = _service(provider)

    chunks = list(service.stream(prompt="hi"))

    assert len(chunks) == 1
    assert chunks[0].done is True
    assert chunks[0].delta == "single shot"


def test_service_status_is_delegated() -> None:
    provider = FakeProvider("ollama", configured=True)
    service = _service(provider)

    statuses = service.status(probe=False)

    assert [s.provider for s in statuses] == ["ollama"]


def test_describe_contains_no_credentials() -> None:
    service = _service(FakeProvider("ollama"), default_model="m")

    summary = service.describe()

    assert summary["enabled"] is True
    assert summary["default_model"] == "m"
    assert "api_key" not in summary
    assert "secret" not in str(summary).lower()
