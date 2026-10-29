"""Explicit long-term memory storage.

Nothing is ever saved automatically: creating a memory requires an explicit
`save()` call by the caller. Retrieved memory is data for the AI - it is
never treated as instructions.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from robanayt_ai.memory.database import Database
from robanayt_ai.memory.errors import MemoryValidationError, SavedMemoryNotFoundError
from robanayt_ai.memory.models import (
    DEFAULT_CATEGORY,
    SavedMemory,
    decode_json,
    encode_json,
    new_id,
    validate_tags,
    validate_text,
)

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persistence for explicitly saved memories."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def save(
        self,
        content: str,
        *,
        category: str = DEFAULT_CATEGORY,
        source: str | None = None,
        importance: int | None = None,
        tags: Sequence[str] | None = None,
        memory_id: str | None = None,
        now: float | None = None,
    ) -> SavedMemory:
        """Save one memory explicitly. Returns the stored memory."""
        timestamp = time.time() if now is None else float(now)
        memory = SavedMemory(
            id=memory_id or new_id("mem"),
            content=validate_text(content),
            created_at=timestamp,
            updated_at=timestamp,
            category=category,
            source=source,
            importance=importance,
            tags=validate_tags(tags),
        )
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO memories "
                "(id, content, category, created_at, updated_at, source, importance, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.content,
                    memory.category,
                    memory.created_at,
                    memory.updated_at,
                    memory.source,
                    memory.importance,
                    encode_json(list(memory.tags)),
                ),
            )
            if self._db.fts_enabled:
                conn.execute(
                    "INSERT INTO memories_fts (content, memory_id, category) VALUES (?, ?, ?)",
                    (memory.content, memory.id, memory.category),
                )
        logger.debug("memory saved: %s", memory.id)
        return memory

    def get(self, memory_id: str) -> SavedMemory | None:
        row = self._db.connection().execute(
            "SELECT id, content, category, created_at, updated_at, source, importance, tags "
            "FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return _row_to_memory(row) if row is not None else None

    def require(self, memory_id: str) -> SavedMemory:
        memory = self.get(memory_id)
        if memory is None:
            raise SavedMemoryNotFoundError(f"saved memory not found: {memory_id}")
        return memory

    def list(
        self,
        *,
        category: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[SavedMemory, ...]:
        """List memories, newest first, with optional category/tag filters."""
        if limit <= 0:
            return ()
        wanted_tags = validate_tags(tags)
        sql = (
            "SELECT id, content, category, created_at, updated_at, source, importance, tags "
            "FROM memories"
        )
        params: list[Any] = []
        conditions: list[str] = []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if wanted_tags and self._db.json_enabled:
            for tag in wanted_tags:
                conditions.append(
                    "EXISTS (SELECT 1 FROM json_each(memories.tags) WHERE json_each.value = ?)"
                )
                params.append(tag)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY updated_at DESC, id ASC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(max(0, offset))])

        rows = self._db.connection().execute(sql, params).fetchall()
        memories = [_row_to_memory(row) for row in rows]

        if wanted_tags and not self._db.json_enabled:
            # Fallback filtering when JSON1 table functions are unavailable.
            memories = [
                memory for memory in memories if set(wanted_tags).issubset(set(memory.tags))
            ]
        return tuple(memories)

    def count(self, *, category: str | None = None) -> int:
        if category:
            row = self._db.connection().execute(
                "SELECT count(*) FROM memories WHERE category = ?", (category,)
            ).fetchone()
        else:
            row = self._db.connection().execute(
                "SELECT count(*) FROM memories"
            ).fetchone()
        return int(row[0]) if row else 0

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: str | None = None,
        source: str | None = None,
        importance: int | None = None,
        tags: Sequence[str] | None = None,
        now: float | None = None,
    ) -> SavedMemory:
        """Update selected fields; unspecified fields keep their value."""
        if all(item is None for item in (content, category, source, importance, tags)):
            raise MemoryValidationError("update() requires at least one field")

        current = self.require(memory_id)
        timestamp = time.time() if now is None else float(now)
        updated = SavedMemory(
            id=current.id,
            content=validate_text(content) if content is not None else current.content,
            created_at=current.created_at,
            updated_at=timestamp,
            category=category or current.category,
            source=source if source is not None else current.source,
            importance=importance if importance is not None else current.importance,
            tags=validate_tags(tags) if tags is not None else current.tags,
        )

        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE memories SET content = ?, category = ?, updated_at = ?, "
                "source = ?, importance = ?, tags = ? WHERE id = ?",
                (
                    updated.content,
                    updated.category,
                    updated.updated_at,
                    updated.source,
                    updated.importance,
                    encode_json(list(updated.tags)),
                    updated.id,
                ),
            )
            if self._db.fts_enabled:
                conn.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?", (updated.id,)
                )
                conn.execute(
                    "INSERT INTO memories_fts (content, memory_id, category) VALUES (?, ?, ?)",
                    (updated.content, updated.id, updated.category),
                )
        logger.debug("memory updated: %s", memory_id)
        return updated

    def delete(self, memory_id: str) -> bool:
        if self.get(memory_id) is None:
            return False
        with self._db.transaction() as conn:
            if self._db.fts_enabled:
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        logger.debug("memory deleted: %s", memory_id)
        return True

    def clear_all(self, *, confirm: bool = False) -> int:
        """Delete every saved memory - requires explicit confirmation."""
        if confirm is not True:
            raise MemoryValidationError(
                "clear_all() deletes every saved memory; pass confirm=True explicitly"
            )
        removed = self.count()
        with self._db.transaction() as conn:
            if self._db.fts_enabled:
                conn.execute("DELETE FROM memories_fts")
            conn.execute("DELETE FROM memories")
        logger.info("all saved memories cleared (%s entries)", removed)
        return removed


def _row_to_memory(row: Any) -> SavedMemory:
    tags = decode_json(row["tags"], [])
    if not isinstance(tags, list):
        tags = []
    importance = row["importance"]
    return SavedMemory(
        id=str(row["id"]),
        content=str(row["content"]),
        category=str(row["category"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        source=None if row["source"] is None else str(row["source"]),
        importance=None if importance is None else int(importance),
        tags=tuple(str(tag) for tag in tags),
    )


def memory_from_mapping(data: Mapping[str, Any]) -> SavedMemory:
    """Build a `SavedMemory` from a plain mapping (import/tooling helper)."""
    return SavedMemory(
        id=str(data.get("id") or new_id("mem")),
        content=data.get("content", ""),
        category=str(data.get("category") or DEFAULT_CATEGORY),
        created_at=float(data.get("created_at") or time.time()),
        updated_at=float(data.get("updated_at") or time.time()),
        source=data.get("source"),
        importance=data.get("importance"),
        tags=validate_tags(data.get("tags")),
    )
