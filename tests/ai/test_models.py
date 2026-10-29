"""AI data models: roles, validation, normalisation."""

from __future__ import annotations

import pytest

from jarvis.ai.errors import AIRequestError
from jarvis.ai.models import (
    AIChunk,
    AIRequest,
    AIResponse,
    AIMessage,
    ProviderStatus,
    Usage,
)


def test_message_helpers_use_valid_roles() -> None:
    assert AIMessage.system("s").role == "system"
    assert AIMessage.user("u").role == "user"
    assert AIMessage.assistant("a").role == "assistant"
    assert AIMessage.tool("t").role == "tool"


def test_invalid_role_is_rejected() -> None:
    with pytest.raises(AIRequestError, match="invalid message role"):
        AIMessage(role="wizard", content="hi")


def test_message_as_dict() -> None:
    assert AIMessage.user("hello").as_dict() == {"role": "user", "content": "hello"}


def test_request_coerces_messages_to_tuple() -> None:
    request = AIRequest(messages=[AIMessage.user("hi")])
    assert isinstance(request.messages, tuple)
    assert len(request.messages) == 1


def test_request_validation_requires_messages() -> None:
    with pytest.raises(AIRequestError, match="no messages"):
        AIRequest().validate()


def test_request_validation_rejects_bad_parameters() -> None:
    base = AIRequest(messages=[AIMessage.user("hi")])

    with pytest.raises(AIRequestError, match="temperature"):
        AIRequest(messages=base.messages, temperature=5.0).validate()

    with pytest.raises(AIRequestError, match="max_tokens"):
        AIRequest(messages=base.messages, max_tokens=0).validate()

    with pytest.raises(AIRequestError, match="timeout"):
        AIRequest(messages=base.messages, timeout=0).validate()


def test_request_with_overrides_fills_defaults_only() -> None:
    request = AIRequest(messages=[AIMessage.user("hi")])

    filled = request.with_overrides(model="m1", timeout=5.0)
    assert filled.model == "m1"
    assert filled.timeout == 5.0

    kept = AIRequest(messages=[AIMessage.user("hi")], model="mine").with_overrides(
        model="other"
    )
    assert kept.model == "mine"


def test_requested_provider_from_metadata() -> None:
    request = AIRequest(messages=[AIMessage.user("hi")], metadata={"provider": "ollama"})
    assert request.requested_provider() == "ollama"
    assert AIRequest(messages=[AIMessage.user("hi")]).requested_provider() is None


def test_response_and_usage_shapes() -> None:
    response = AIResponse(
        content="hi",
        model="m",
        provider="p",
        finish_reason="stop",
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    assert response.text == "hi"
    assert response.usage.as_dict()["total_tokens"] == 3


def test_chunk_final_helper() -> None:
    chunk = AIChunk.final(model="m", provider="p")
    assert chunk.done is True
    assert chunk.finish_reason == "stop"
    assert chunk.delta == ""


def test_provider_status_ok_flag() -> None:
    assert ProviderStatus(provider="x", configured=True, available=True).ok is True
    assert ProviderStatus(provider="x", configured=True, available=False).ok is False
    assert ProviderStatus(provider="x").as_dict()["provider"] == "x"
