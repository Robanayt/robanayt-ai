"""Context preparation for the Phase 2 AI layer."""

from __future__ import annotations

import pytest

from jarvis.ai.models import AIRequest, AIMessage
from jarvis.core.config import MemoryConfig
from jarvis.memory.service import MemoryService


@pytest.fixture()
def service(tmp_path) -> MemoryService:
    return MemoryService(MemoryConfig(directory=str(tmp_path / "memory")))


def _conversation_with_messages(service: MemoryService, count: int, prefix: str = "m") -> str:
    conversation = service.conversations.create_conversation("ctx", now=1.0)
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        service.conversations.add_message(
            conversation.id, role, f"{prefix}{index}", created_at=10.0 + index
        )
    return conversation.id


def test_context_limits_message_count(service) -> None:
    conversation_id = _conversation_with_messages(service, 10)

    context = service.prepare_context(conversation_id, max_messages=4)

    assert len(context.messages) == 4
    assert [m.content for m in context.messages] == ["m6", "m7", "m8", "m9"]
    assert context.truncated is True


def test_context_respects_character_budget(service) -> None:
    conversation_id = service.conversations.create_conversation("budget", now=1.0).id
    service.conversations.add_message(conversation_id, "user", "x" * 50, created_at=1.0)
    service.conversations.add_message(conversation_id, "assistant", "y" * 50, created_at=2.0)
    service.conversations.add_message(conversation_id, "user", "z" * 50, created_at=3.0)

    context = service.prepare_context(conversation_id, max_chars=120)

    assert context.truncated is True
    assert len(context.messages) == 2
    assert [m.content for m in context.messages] == ["y" * 50, "z" * 50]
    assert context.chars == 100


def test_newest_message_is_always_kept(service) -> None:
    conversation_id = service.conversations.create_conversation("newest", now=1.0).id
    service.conversations.add_message(conversation_id, "user", "old", created_at=1.0)
    service.conversations.add_message(conversation_id, "user", "newest entry", created_at=2.0)

    context = service.prepare_context(conversation_id, max_chars=5)

    assert [m.content for m in context.messages] == ["newest entry"]
    assert context.truncated is True


def test_without_conversation_id_context_is_empty(service) -> None:
    context = service.prepare_context(None, system_prompt="sys")

    assert context.messages == ()
    assert context.truncated is False
    assert [m.role for m in context.to_ai_messages()] == ["system"]


def test_system_prompt_stays_separate_from_messages(service) -> None:
    conversation_id = _conversation_with_messages(service, 2)

    context = service.prepare_context(conversation_id, system_prompt="you are JARVIS")

    assert context.system_prompt == "you are JARVIS"
    assert all(m.role != "system" for m in context.messages)
    messages = context.to_ai_messages()
    assert messages[0].role == "system"
    assert messages[0].content == "you are JARVIS"


def test_memories_are_only_included_when_supplied(service) -> None:
    conversation_id = _conversation_with_messages(service, 1)
    service.remember("user likes tea")

    without_snippets = service.prepare_context(conversation_id, system_prompt="sys")
    with_snippets = service.prepare_context(
        conversation_id, system_prompt="sys", memory_snippets=("likes tea",)
    )

    assert without_snippets.memory_snippets == ()
    assert "likes tea" not in without_snippets.to_ai_messages()[0].content
    assert "likes tea" in with_snippets.to_ai_messages()[0].content


def test_select_memory_snippets_is_explicit(service) -> None:
    service.remember("user likes tea", category="preferences")

    assert service.select_memory_snippets("tea") != ()
    assert service.select_memory_snippets("nothingmatches") == ()


def test_request_is_compatible_with_phase2(service) -> None:
    conversation_id = _conversation_with_messages(service, 3)

    context = service.prepare_context(conversation_id, system_prompt="sys")
    request = context.to_ai_request(model="test-model", temperature=0.2)

    assert isinstance(request, AIRequest)
    assert all(isinstance(m, AIMessage) for m in request.messages)
    request.validate()
    assert request.model == "test-model"
    assert request.temperature == 0.2


def test_default_limit_comes_from_configuration(tmp_path) -> None:
    config = MemoryConfig(directory=str(tmp_path / "memory"), max_recent_messages=3)
    service = MemoryService(config)
    conversation_id = _conversation_with_messages(service, 8)

    context = service.prepare_context(conversation_id)

    assert len(context.messages) == 3
