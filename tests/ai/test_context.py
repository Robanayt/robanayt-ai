"""Conversation context: system prompt, rolling history, request building."""

from __future__ import annotations

import pytest

from jarvis.ai.context import DEFAULT_MAX_MESSAGES, ConversationContext
from jarvis.ai.errors import AIRequestError
from jarvis.ai.models import AIMessage


def test_context_starts_empty() -> None:
    context = ConversationContext()

    assert context.system_prompt is None
    assert context.messages == ()
    assert len(context) == 0
    assert context.max_messages == DEFAULT_MAX_MESSAGES


def test_add_messages_in_order() -> None:
    context = ConversationContext()
    context.add_user("hello")
    context.add_assistant("hi there")

    assert [m.role for m in context.messages] == ["user", "assistant"]
    assert len(context) == 2


def test_history_limit_trims_oldest() -> None:
    context = ConversationContext(max_messages=2)
    for index in range(5):
        context.add_user(f"m{index}")

    assert [m.content for m in context.messages] == ["m3", "m4"]


def test_history_returns_tail_slice() -> None:
    context = ConversationContext()
    context.add_user("one")
    context.add_user("two")
    context.add_user("three")

    assert [m.content for m in context.history(limit=2)] == ["two", "three"]
    assert context.history(limit=0) == ()


def test_system_prompt_leads_request_messages() -> None:
    context = ConversationContext("you are jarvis")
    context.add_user("hi")

    messages = context.to_messages()

    assert messages[0] == AIMessage.system("you are jarvis")
    assert messages[1] == AIMessage.user("hi")


def test_set_system_and_clear() -> None:
    context = ConversationContext()
    context.set_system("new system")
    context.add_user("x")
    context.clear()

    assert context.system_prompt == "new system"
    assert len(context) == 0


def test_to_request_builds_valid_request() -> None:
    context = ConversationContext("system")
    context.add_user("question")

    request = context.to_request(model="m", temperature=0.2, max_tokens=64, timeout=5.0)

    request.validate()
    assert request.model == "m"
    assert len(request.messages) == 2
    assert request.temperature == 0.2
    assert request.max_tokens == 64
    assert request.timeout == 5.0


def test_empty_context_request_is_invalid() -> None:
    request = ConversationContext().to_request()

    with pytest.raises(AIRequestError):
        request.validate()


def test_add_requires_ai_message() -> None:
    context = ConversationContext()

    with pytest.raises(TypeError):
        context.add("not a message")  # type: ignore[arg-type]


def test_negative_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        ConversationContext(max_messages=0)


def test_extend_adds_multiple_messages() -> None:
    context = ConversationContext(max_messages=10)
    context.extend([AIMessage.user("a"), AIMessage.assistant("b")])

    assert len(context) == 2
