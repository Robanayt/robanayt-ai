"""MemoryService orchestration, privacy and AI integration."""

from __future__ import annotations

import logging
import threading

import pytest
from tests.ai.fakes import FakeProvider

from jarvis.ai.router import ProviderRouter
from jarvis.ai.service import AIService
from jarvis.core.config import MemoryConfig
from jarvis.memory.errors import (
    MemoryDisabledError,
    MemoryExportError,
    MemoryImportError,
)
from jarvis.memory.service import MemoryService, safe_output_path


def _config(tmp_path, **kwargs) -> MemoryConfig:
    base = {"directory": str(tmp_path / "memory")}
    base.update(kwargs)
    return MemoryConfig(**base)


def test_memory_is_usable_from_another_thread(tmp_path) -> None:
    """Regression: a turn writes on a worker thread, the UI reads on its own.

    Before `check_same_thread=False` the second thread raised
    "SQLite objects created in a thread can only be used in that same thread".
    """
    service = MemoryService(_config(tmp_path))
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            conversation = service.create_conversation("worker thread")
            service.add_message(conversation.id, "user", "привет")
        except BaseException as exc:  # noqa: BLE001 - reported to the assertion
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert errors == []
    conversations = service.conversations.list_conversations(limit=10)
    assert len(conversations) == 1
    assert service.conversations.get_messages(conversations[0].id)[0].content == "привет"
    service.close()


def test_concurrent_reads_and_writes_do_not_corrupt_the_store(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    service.create_conversation("базовый")
    errors: list[BaseException] = []

    def writer(index: int) -> None:
        try:
            conversation = service.create_conversation(f"диалог {index}")
            service.add_message(conversation.id, "user", f"сообщение {index}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(25):
                service.conversations.list_conversations(limit=5)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
    threads.append(threading.Thread(target=reader))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(service.conversations.list_conversations(limit=50)) == 9
    service.close()


def _ai(reply: str = "stub answer from fake provider") -> AIService:
    provider = FakeProvider("fake", reply=reply)
    return AIService(ProviderRouter([provider]), default_model="fake-model")


def test_construction_is_lazy(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))

    assert service.database_path.is_file() is False
    assert service.is_initialized is False
    assert service.is_open is False


def test_first_operation_creates_the_database(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))

    service.create_conversation("first", now=1.0)

    assert service.database_path.is_file() is True
    assert service.is_initialized is True
    service.close()


def test_disabled_service_refuses_every_operation(tmp_path) -> None:
    service = MemoryService(_config(tmp_path, enabled=False))

    assert service.enabled is False
    with pytest.raises(MemoryDisabledError):
        service.conversations
    with pytest.raises(MemoryDisabledError):
        service.memories
    with pytest.raises(MemoryDisabledError):
        service.prepare_context("conv_1")
    with pytest.raises(MemoryDisabledError):
        service.stats()
    # Diagnostics still work and never touch the database.
    assert service.describe()["enabled"] is False


def test_stats_report_counts(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("stats", now=1.0)
    service.add_message(conversation.id, "user", "hello")
    service.remember("user likes tea")

    stats = service.stats()

    assert stats.conversations == 1
    assert stats.messages == 1
    assert stats.memories == 1
    assert stats.database_path == str(service.database_path)
    assert stats.search_backend in {"fts5", "like"}
    service.close()


def test_data_survives_separate_service_instances(tmp_path) -> None:
    first = MemoryService(_config(tmp_path))
    conversation = first.create_conversation("persist", now=1.0)
    first.add_message(conversation.id, "user", "remember me")
    first.close()

    second = MemoryService(_config(tmp_path))
    loaded = second.conversations.get_conversation(conversation.id)

    assert loaded is not None
    assert second.conversations.count_messages(conversation.id) == 1
    second.close()


def test_run_turn_persists_both_sides(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("turn", now=1.0)

    result = service.run_turn(_ai(), conversation.id, "hello there")

    assert result.response.content == "stub answer from fake provider"
    messages = service.conversations.get_messages(conversation.id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hello there"),
        ("assistant", "stub answer from fake provider"),
    ]
    assert result.user_message is not None
    assert result.assistant_message is not None
    service.close()


def test_run_turn_can_skip_persistence(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("nopersist", now=1.0)

    result = service.run_turn(_ai(), conversation.id, "hello", persist=False)

    assert result.persisted is False
    assert service.conversations.count_messages(conversation.id) == 0
    service.close()


def test_run_turn_uses_stored_context(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("ctx", now=1.0)
    service.add_message(conversation.id, "user", "earlier question")

    result = service.run_turn(_ai(), conversation.id, "follow up", persist=False)

    request = result.response  # FakeProvider echoes nothing; check via provider below
    assert request is not None
    messages = service.conversations.get_messages(conversation.id)
    assert [m.content for m in messages] == ["earlier question"]
    service.close()


def test_run_turn_request_contains_history(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("history", now=1.0)
    service.add_message(conversation.id, "user", "earlier question")

    provider = FakeProvider("fake", reply="ok")
    ai = AIService(ProviderRouter([provider]), default_model="fake-model")
    service.run_turn(ai, conversation.id, "follow up", persist=False, system_prompt="sys")

    sent = provider.calls[0]
    assert [m.role for m in sent.messages] == ["system", "user", "user"]
    assert [m.content for m in sent.messages][1:] == ["earlier question", "follow up"]
    service.close()


def test_empty_assistant_response_is_not_persisted(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("empty", now=1.0)

    result = service.run_turn(_ai(reply=""), conversation.id, "hello")

    assert result.assistant_message is None
    assert service.conversations.count_messages(conversation.id) == 1
    service.close()


def test_logs_never_contain_private_content(tmp_path, caplog) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("private", now=1.0)

    with caplog.at_level(logging.DEBUG):
        service.add_message(conversation.id, "user", "SECRET-SENTENCE-42")
        service.remember("SECRET-MEMORY-42")
        service.stats()

    assert "SECRET-SENTENCE-42" not in caplog.text
    assert "SECRET-MEMORY-42" not in caplog.text
    service.close()


def test_export_and_import_round_trip(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("export", now=1.0)
    service.add_message(conversation.id, "user", "exported text")
    target = tmp_path / "out" / "conversation.json"

    path = service.export_conversation_to_file(conversation.id, target)
    service.conversations.delete_conversation(conversation.id)
    restored = service.import_conversation_from_file(path)

    assert path.is_file()
    assert restored.id == conversation.id
    assert [m.content for m in service.conversations.get_messages(restored.id)] == [
        "exported text"
    ]
    service.close()


def test_export_refuses_overwrite_and_traversal(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    conversation = service.create_conversation("guard", now=1.0)
    target = tmp_path / "exists.json"
    service.export_conversation_to_file(conversation.id, target)

    with pytest.raises(MemoryExportError, match="overwrite"):
        service.export_conversation_to_file(conversation.id, target)
    with pytest.raises(MemoryExportError, match=r"\.\."):
        safe_output_path(tmp_path / ".." / "evil.json")
    with pytest.raises(MemoryImportError):
        service.import_conversation_from_file(tmp_path / "missing.json")
    service.close()


def test_remember_and_recall(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))

    service.remember("user prefers dark mode", category="preferences", tags=("ui",))
    results = service.recall("dark mode")

    assert len(results) == 1
    assert results[0].kind == "memory"
    service.close()


def test_close_releases_connection(tmp_path) -> None:
    service = MemoryService(_config(tmp_path))
    service.create_conversation("x", now=1.0)

    service.close()

    assert service.is_open is False
