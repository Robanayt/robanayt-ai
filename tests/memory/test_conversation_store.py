"""Conversation store: persistence, ordering, pagination, export/import."""

from __future__ import annotations

import pytest

from jarvis.memory.conversation_store import ConversationStore
from jarvis.memory.database import Database
from jarvis.memory.errors import (
    ConversationNotFoundError,
    MemoryImportError,
    MemoryValidationError,
)
from jarvis.memory.models import Conversation


@pytest.fixture()
def store(tmp_path) -> ConversationStore:
    return ConversationStore(Database(tmp_path / "jarvis.db"))


def test_create_and_get(store) -> None:
    conversation = store.create_conversation("first", now=100.0)

    loaded = store.get_conversation(conversation.id)

    assert loaded == conversation
    assert loaded.title == "first"


def test_list_is_paginated_and_newest_first(store) -> None:
    for index in range(5):
        store.create_conversation(f"c{index}", now=100.0 + index)

    first_page = store.list_conversations(limit=2, offset=0)
    second_page = store.list_conversations(limit=2, offset=2)

    assert [c.title for c in first_page] == ["c4", "c3"]
    assert [c.title for c in second_page] == ["c2", "c1"]
    assert store.list_conversations(limit=0) == ()
    assert store.count_conversations() == 5


def test_messages_keep_chronological_order_with_equal_timestamps(store) -> None:
    conversation = store.create_conversation("order", now=1.0)
    for index, text in enumerate(["a", "b", "c", "d"]):
        store.add_message(conversation.id, "user", text, created_at=5.0)

    messages = store.get_messages(conversation.id)

    assert [m.content for m in messages] == ["a", "b", "c", "d"]
    assert [m.seq for m in messages] == [0, 1, 2, 3]
    assert store.count_messages(conversation.id) == 4


def test_recent_messages_returns_newest_oldest_first(store) -> None:
    conversation = store.create_conversation("recent", now=1.0)
    for index in range(6):
        store.add_message(conversation.id, "user", f"m{index}", created_at=10.0 + index)

    recent = store.recent_messages(conversation.id, 3)

    assert [m.content for m in recent] == ["m3", "m4", "m5"]
    assert store.recent_messages(conversation.id, 0) == ()


def test_get_messages_pagination(store) -> None:
    conversation = store.create_conversation("page", now=1.0)
    for index in range(4):
        store.add_message(conversation.id, "user", f"m{index}", created_at=1.0 + index)

    assert [m.content for m in store.get_messages(conversation.id, limit=2)] == ["m0", "m1"]
    assert [m.content for m in store.get_messages(conversation.id, limit=2, offset=2)] == [
        "m2",
        "m3",
    ]
    assert store.get_messages(conversation.id, limit=0) == ()


def test_add_message_rejects_unknown_conversation(store) -> None:
    with pytest.raises(ConversationNotFoundError):
        store.add_message("conv_missing", "user", "hi")


def test_add_message_validates_metadata(store) -> None:
    conversation = store.create_conversation(now=1.0)

    with pytest.raises(MemoryValidationError, match="credential"):
        store.add_message(conversation.id, "user", "hi", metadata={"api_key": "x"})


def test_rename_keeps_messages(store) -> None:
    conversation = store.create_conversation("old", now=1.0)
    store.add_message(conversation.id, "user", "hi", created_at=1.0)

    renamed = store.rename_conversation(conversation.id, "new")

    assert renamed.title == "new"
    assert store.count_messages(conversation.id) == 1
    assert store.get_conversation(conversation.id).title == "new"


def test_delete_removes_messages_and_conversation(store) -> None:
    conversation = store.create_conversation(now=1.0)
    store.add_message(conversation.id, "user", "hi", created_at=1.0)

    assert store.delete_conversation(conversation.id) is True
    assert store.get_conversation(conversation.id) is None
    assert store.count_messages(conversation.id) == 0
    assert store.delete_conversation("conv_missing") is False


def test_export_import_round_trip(store) -> None:
    conversation = store.create_conversation("trip", now=1.0)
    store.add_message(conversation.id, "user", "question", created_at=1.0)
    store.add_message(conversation.id, "assistant", "answer", created_at=2.0)

    payload = store.export_conversation(conversation.id)
    store.delete_conversation(conversation.id)

    restored = store.import_conversation(payload)

    assert restored.id == conversation.id
    messages = store.get_messages(restored.id)
    assert [m.content for m in messages] == ["question", "answer"]
    assert [m.role for m in messages] == ["user", "assistant"]


def test_export_payload_is_json_safe(store) -> None:
    import json

    conversation = store.create_conversation("json", now=1.0)
    store.add_message(conversation.id, "user", "hi", created_at=1.0)

    payload = store.export_conversation(conversation.id)

    assert json.loads(json.dumps(payload))["conversation"]["id"] == conversation.id


def test_import_never_overwrites_existing_ids(store) -> None:
    original = store.create_conversation("original", now=1.0)
    store.add_message(original.id, "user", "original text", created_at=1.0)
    payload = store.export_conversation(original.id)

    duplicate = store.import_conversation(payload)

    assert duplicate.id != original.id
    assert store.count_conversations() == 2
    assert [m.content for m in store.get_messages(original.id)] == ["original text"]
    assert [m.content for m in store.get_messages(duplicate.id)] == ["original text"]


def test_import_rejects_bad_version(store) -> None:
    conversation = store.create_conversation(now=1.0)
    payload = store.export_conversation(conversation.id)
    payload["export_version"] = 99

    with pytest.raises(MemoryImportError, match="unsupported export_version"):
        store.import_conversation(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("export_version"),
        lambda payload: payload.update(conversation="not an object"),
        lambda payload: payload.update(messages="not a list"),
        lambda payload: payload.update(messages=[{"role": "wizard", "content": "x"}]),
        lambda payload: payload.update(messages=[{"role": "user", "content": "  "}]),
        lambda payload: payload.update(messages=[{"role": "user", "content": "x" * 40000}]),
        lambda payload: payload.update(
            messages=[{"role": "user", "content": "x", "metadata": {"api_key": "s"}}]
        ),
    ],
)
def test_import_rejects_malformed_payloads(store, mutate) -> None:
    conversation = store.create_conversation(now=1.0)
    payload = store.export_conversation(conversation.id)
    mutate(payload)

    with pytest.raises(MemoryImportError):
        store.import_conversation(payload)


def test_import_rejects_non_mapping(store) -> None:
    with pytest.raises(MemoryImportError):
        store.import_conversation(["not", "an", "object"])


def test_imported_content_is_data_not_code(store) -> None:
    dangerous = "__import__('os').system('echo hi')"
    payload = {
        "export_version": 1,
        "conversation": Conversation(
            id="conv_imported", title="t", created_at=1.0, updated_at=1.0
        ).to_dict(),
        "messages": [{"role": "user", "content": dangerous, "created_at": 1.0}],
    }

    conversation = store.import_conversation(payload)
    stored = store.get_messages(conversation.id)[0]

    assert stored.content == dangerous
    assert isinstance(stored.content, str)
