"""Saved-memory store: explicit CRUD, filtering, persistence."""

from __future__ import annotations

import pytest

from jarvis.memory.database import Database
from jarvis.memory.errors import MemoryValidationError, SavedMemoryNotFoundError
from jarvis.memory.memory_store import MemoryStore


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    return MemoryStore(Database(tmp_path / "jarvis.db"))


def test_save_and_get(store) -> None:
    memory = store.save("user prefers dark mode", category="preferences", now=1.0)

    loaded = store.get(memory.id)

    assert loaded == memory
    assert loaded.content == "user prefers dark mode"
    assert loaded.category == "preferences"


def test_require_raises_for_missing(store) -> None:
    with pytest.raises(SavedMemoryNotFoundError):
        store.require("mem_missing")


def test_list_newest_first_with_pagination(store) -> None:
    for index in range(4):
        store.save(f"memory {index}", now=10.0 + index)

    page = store.list(limit=2)
    assert [m.content for m in page] == ["memory 3", "memory 2"]
    assert [m.content for m in store.list(limit=2, offset=2)] == ["memory 1", "memory 0"]
    assert store.list(limit=0) == ()


def test_filter_by_category(store) -> None:
    store.save("a", category="preferences", now=1.0)
    store.save("b", category="facts", now=2.0)

    assert [m.content for m in store.list(category="facts")] == ["b"]
    assert store.count(category="facts") == 1
    assert store.count() == 2


def test_filter_by_tags(store) -> None:
    store.save("a", tags=("ui", "theme"), now=1.0)
    store.save("b", tags=("ui",), now=2.0)
    store.save("c", tags=("other",), now=3.0)

    both = store.list(tags=("ui", "theme"))
    assert [m.content for m in both] == ["a"]
    assert len(store.list(tags=("ui",))) == 2


def test_update_changes_only_given_fields(store) -> None:
    memory = store.save("old text", category="facts", importance=2, now=1.0)

    updated = store.update(memory.id, content="new text", now=5.0)

    assert updated.content == "new text"
    assert updated.category == "facts"
    assert updated.importance == 2
    assert updated.created_at == 1.0
    assert updated.updated_at == 5.0


def test_update_requires_a_field(store) -> None:
    memory = store.save("x", now=1.0)

    with pytest.raises(MemoryValidationError, match="at least one field"):
        store.update(memory.id)


def test_update_missing_raises(store) -> None:
    with pytest.raises(SavedMemoryNotFoundError):
        store.update("mem_missing", content="x")


def test_delete(store) -> None:
    memory = store.save("x", now=1.0)

    assert store.delete(memory.id) is True
    assert store.get(memory.id) is None
    assert store.delete("mem_missing") is False


def test_clear_all_requires_confirmation(store) -> None:
    store.save("x", now=1.0)

    with pytest.raises(MemoryValidationError, match="confirm=True"):
        store.clear_all()

    assert store.count() == 1

    removed = store.clear_all(confirm=True)
    assert removed == 1
    assert store.count() == 0


def test_persistence_across_connections(tmp_path) -> None:
    path = tmp_path / "jarvis.db"
    first = MemoryStore(Database(path))
    memory = first.save("persisted memory", category="facts", tags=("a",), now=1.0)
    first._db.close()

    second = MemoryStore(Database(path))
    loaded = second.get(memory.id)

    assert loaded is not None
    assert loaded.content == "persisted memory"
    assert loaded.tags == ("a",)
    second._db.close()


def test_saved_memory_validation_errors(store) -> None:
    with pytest.raises(MemoryValidationError):
        store.save("   ", now=1.0)
    with pytest.raises(MemoryValidationError, match="importance"):
        store.save("x", importance=42, now=1.0)
