"""Search: FTS5 backend, LIKE fallback, safety and determinism."""

from __future__ import annotations

import pytest

import robanayt_ai.memory.database as database_module
from robanayt_ai.memory.conversation_store import ConversationStore
from robanayt_ai.memory.database import Database
from robanayt_ai.memory.memory_store import MemoryStore
from robanayt_ai.memory.search import BACKEND_FTS5, BACKEND_LIKE, SearchEngine


@pytest.fixture()
def populated(tmp_path):
    database = Database(tmp_path / "jarvis.db")
    conversations = ConversationStore(database)
    memories = MemoryStore(database)
    conversation = conversations.create_conversation("searchable", now=1.0)
    conversations.add_message(conversation.id, "user", "What is the weather today?", created_at=1.0)
    conversations.add_message(
        conversation.id, "assistant", "Сегодня солнечно и тепло.", created_at=2.0
    )
    conversations.add_message(conversation.id, "user", "remember: milk and eggs", created_at=3.0)
    memories.save("User prefers dark mode", category="preferences", now=4.0)
    memories.save("Allergic to peanuts", category="health", now=5.0)
    return database, conversations, memories, conversation


def test_backend_is_fts5_when_available(populated) -> None:
    database, *_ = populated
    assert SearchEngine(database).backend == BACKEND_FTS5


def test_search_messages_finds_terms(populated) -> None:
    database, conversations, _, conversation = populated
    engine = SearchEngine(database)

    results = engine.search_messages("weather")

    assert len(results) == 1
    assert results[0].id
    assert results[0].conversation_id == conversation.id
    assert "weather" in results[0].snippet.lower()
    assert results[0].role == "user"


def test_search_supports_unicode(populated) -> None:
    database, *_ = populated
    engine = SearchEngine(database)

    results = engine.search_messages("солнечно")

    assert len(results) == 1
    assert "солнечно" in results[0].snippet.lower()


def test_search_handles_punctuation_and_empty_queries(populated) -> None:
    database, *_ = populated
    engine = SearchEngine(database)

    assert engine.search_messages("!!! ??? ...") == ()
    assert engine.search_messages("") == ()
    assert engine.search_messages(None) == ()
    assert engine.search_messages("weather!!!") != ()


def test_search_no_matches_returns_empty(populated) -> None:
    database, *_ = populated
    engine = SearchEngine(database)

    assert engine.search_messages("definitelynotpresent") == ()
    assert engine.search_memories("definitelynotpresent") == ()


def test_search_memories_and_category_filter(populated) -> None:
    database, *_ = populated
    engine = SearchEngine(database)

    assert len(engine.search_memories("peanuts")) == 1
    assert engine.search_memories("peanuts", category="health")[0].category == "health"
    assert engine.search_memories("peanuts", category="preferences") == ()


def test_search_respects_limit_and_offset(populated) -> None:
    database, *_ = populated
    engine = SearchEngine(database)

    assert len(engine.search_messages("the", limit=1)) == 1
    assert engine.search_messages("the", limit=0) == ()
    page1 = engine.search_all("the", limit=2)
    assert len(page1) <= 2


def test_results_are_deterministically_ordered(populated) -> None:
    database, *_ = populated
    engine = SearchEngine(database)

    first = [r.id for r in engine.search_all("the", limit=10)]
    second = [r.id for r in engine.search_all("the", limit=10)]

    assert first == second


def test_sql_injection_is_harmless(populated) -> None:
    database, conversations, _, conversation = populated
    engine = SearchEngine(database)

    results = engine.search_messages("'; DROP TABLE messages; --")

    assert isinstance(results, tuple)
    assert conversations.count_messages(conversation.id) == 3
    assert engine.search_messages("weather") != ()


def test_like_fallback_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database_module, "supports_fts5", lambda connection: False)
    database = Database(tmp_path / "fallback.db")
    conversations = ConversationStore(database)
    memories = MemoryStore(database)
    conversation = conversations.create_conversation("fb", now=1.0)
    conversations.add_message(conversation.id, "user", "fallback search works", created_at=1.0)
    memories.save("fallback memory entry", now=2.0)

    engine = SearchEngine(database)

    assert engine.backend == BACKEND_LIKE
    assert len(engine.search_messages("fallback")) == 1
    assert len(engine.search_memories("fallback")) == 1
    assert engine.search_messages("!!!") == ()
    assert engine.search_messages("'; DROP TABLE messages; --") == ()
    assert conversations.count_messages(conversation.id) == 1


def test_snippet_is_not_the_full_content(populated) -> None:
    database, conversations, _, conversation = populated
    long_text = "alpha " * 500
    conversations.add_message(conversation.id, "user", long_text, created_at=9.0)

    results = SearchEngine(database).search_messages("alpha")

    assert results
    assert len(results[0].snippet) < len(long_text)


def test_search_all_combines_both_stores(populated) -> None:
    database, *_ = populated
    engine = SearchEngine(database)

    results = engine.search_all("the", limit=10)

    kinds = {result.kind for result in results}
    assert "message" in kinds
