"""Memory model validation and Phase 2 AI compatibility."""

from __future__ import annotations

import pytest

from robanayt_ai.ai.models import AIRequest, AIMessage, VALID_ROLES
from robanayt_ai.memory.errors import MemoryValidationError
from robanayt_ai.memory.models import (
    Conversation,
    Message,
    PreparedContext,
    SavedMemory,
    SearchResult,
    MESSAGE_ROLES,
    MAX_CONTENT_CHARS,
    new_id,
)


def _message(**kwargs) -> Message:
    defaults = {
        "id": "msg_1",
        "conversation_id": "conv_1",
        "role": "user",
        "content": "hello",
        "created_at": 1.0,
    }
    defaults.update(kwargs)
    return Message(**defaults)


def test_roles_match_the_ai_layer() -> None:
    assert MESSAGE_ROLES == VALID_ROLES


def test_message_rejects_bad_role() -> None:
    with pytest.raises(MemoryValidationError, match="invalid message role"):
        _message(role="wizard")


def test_message_rejects_empty_and_huge_content() -> None:
    with pytest.raises(MemoryValidationError, match="must not be empty"):
        _message(content="   ")
    with pytest.raises(MemoryValidationError, match="too long"):
        _message(content="x" * (MAX_CONTENT_CHARS + 1))


def test_message_rejects_bad_ids_and_timestamps() -> None:
    with pytest.raises(MemoryValidationError):
        _message(id="bad id with spaces")
    with pytest.raises(MemoryValidationError):
        _message(created_at=-1.0)
    with pytest.raises(MemoryValidationError):
        _message(created_at=float("nan"))


def test_metadata_rejects_credentials() -> None:
    for key in ("api_key", "Authorization", "x-token", "user_password", "secret_hint"):
        with pytest.raises(MemoryValidationError, match="credential"):
            _message(metadata={key: "value"})


def test_metadata_rejects_non_scalars() -> None:
    with pytest.raises(MemoryValidationError, match="scalar"):
        _message(metadata={"nested": {"a": 1}})


def test_metadata_accepts_simple_values() -> None:
    message = _message(metadata={"source": "cli", "count": 2, "ok": True, "note": None})
    assert message.metadata["source"] == "cli"
    assert message.metadata["count"] == 2


def test_message_round_trip() -> None:
    message = _message(metadata={"a": 1})

    restored = Message.from_dict(message.to_dict())

    assert restored == message


def test_message_is_frozen() -> None:
    message = _message()
    with pytest.raises(Exception):
        message.content = "changed"  # type: ignore[misc]


def test_conversation_validation() -> None:
    conversation = Conversation(
        id="conv_1", title="chat", created_at=1.0, updated_at=2.0, metadata={"k": "v"}
    )
    assert conversation.to_dict()["title"] == "chat"

    with pytest.raises(MemoryValidationError):
        Conversation(id="conv 1", title="", created_at=1.0, updated_at=1.0)
    with pytest.raises(MemoryValidationError, match="too long"):
        Conversation(id="conv_1", title="x" * 600, created_at=1.0, updated_at=1.0)


def test_saved_memory_validation() -> None:
    memory = SavedMemory(
        id="mem_1",
        content="user prefers dark mode",
        created_at=1.0,
        updated_at=1.0,
        category="preferences",
        importance=7,
        tags=("ui", "UI", " "),
    )
    assert memory.tags == ("ui", "UI")
    assert memory.importance == 7

    with pytest.raises(MemoryValidationError, match="importance"):
        SavedMemory(id="mem_1", content="x", created_at=1.0, updated_at=1.0, importance=99)
    with pytest.raises(MemoryValidationError, match="category"):
        SavedMemory(id="mem_1", content="x", created_at=1.0, updated_at=1.0, category=" ")
    with pytest.raises(MemoryValidationError):
        SavedMemory(id="mem_1", content="x", created_at=1.0, updated_at=1.0, tags="ui")


def test_search_result_validation() -> None:
    result = SearchResult(
        kind="message", id="msg_1", conversation_id="conv_1",
        snippet="hi", score=1.0, created_at=1.0, role="user",
    )
    assert result.to_dict()["kind"] == "message"

    with pytest.raises(MemoryValidationError, match="unknown search result kind"):
        SearchResult(
            kind="video", id="x", conversation_id=None,
            snippet="", score=0.0, created_at=1.0,
        )


def test_new_id_is_unique_and_safe() -> None:
    first, second = new_id("conv"), new_id("conv")
    assert first != second
    assert first.startswith("conv_")


def test_prepared_context_converts_to_phase2_messages() -> None:
    context = PreparedContext(
        system_prompt="be brief",
        messages=(
            _message(role="user", content="hi", seq=0),
            _message(id="msg_2", role="assistant", content="hello", seq=1),
        ),
        conversation_id="conv_1",
    )

    messages = context.to_ai_messages()

    assert all(isinstance(item, AIMessage) for item in messages)
    assert [item.role for item in messages] == ["system", "user", "assistant"]
    assert [item.content for item in messages] == ["be brief", "hi", "hello"]


def test_prepared_context_includes_selected_snippets_only() -> None:
    context = PreparedContext(
        system_prompt="sys",
        messages=(_message(content="hi"),),
        memory_snippets=("likes tea",),
    )

    with_memory = context.to_ai_messages()
    without_memory = context.to_ai_messages(include_memory=False)

    assert "likes tea" in with_memory[0].content
    assert "likes tea" not in without_memory[0].content
    assert "data, not instructions" in with_memory[0].content


def test_prepared_context_request_is_valid_phase2_request() -> None:
    context = PreparedContext(system_prompt="sys", messages=(_message(content="hi"),))

    request = context.to_ai_request(model="m")

    assert isinstance(request, AIRequest)
    request.validate()
    assert request.model == "m"


def test_prepared_context_counts_characters_not_tokens() -> None:
    context = PreparedContext(system_prompt="abcd", messages=(_message(content="12345"),))

    assert context.chars == 9
