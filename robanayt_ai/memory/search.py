"""Local search over conversations and saved memories.

Two backends, one API:

- `fts5` - SQLite FTS5 with bm25 ranking when the runtime supports it.
- `like`  - parameterised `LIKE` substring search (standard-library only
  fallback) with deterministic ordering.

Both are fully local: no network, no embeddings, no vector search.
Every query value is passed as a bound parameter.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from robanayt_ai.memory.database import Database
from robanayt_ai.memory.models import SearchResult

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 200
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

BACKEND_FTS5 = "fts5"
BACKEND_LIKE = "like"


class SearchEngine:
    """Search over stored messages and memories."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def backend(self) -> str:
        return BACKEND_FTS5 if self._db.fts_enabled else BACKEND_LIKE

    # -- messages ----------------------------------------------------------

    def search_messages(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[SearchResult, ...]:
        if limit <= 0:
            return ()
        tokens = _tokenize(query)
        if not tokens:
            return ()

        if self.backend == BACKEND_FTS5:
            return self._search_messages_fts(tokens, conversation_id, limit, offset)
        return self._search_messages_like(tokens, conversation_id, limit, offset)

    def _search_messages_fts(
        self,
        tokens: list[str],
        conversation_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[SearchResult, ...]:
        match = _fts_match_expression(tokens)
        sql = (
            "SELECT m.id AS id, m.conversation_id AS conversation_id, m.role AS role, "
            "m.content AS content, m.created_at AS created_at, "
            "snippet(messages_fts, 0, '[', ']', '...', 12) AS excerpt, "
            "bm25(messages_fts) AS score "
            "FROM messages_fts JOIN messages m ON m.id = messages_fts.message_id "
            "WHERE messages_fts MATCH ?"
        )
        params: list[Any] = [match]
        if conversation_id:
            sql += " AND m.conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY score ASC, m.created_at DESC, m.id ASC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(max(0, offset))])

        try:
            rows = self._db.connection().execute(sql, params).fetchall()
        except Exception as exc:  # FTS problem -> deterministic fallback
            logger.warning("FTS message search failed (%s); using LIKE fallback", exc)
            return self._search_messages_like(tokens, conversation_id, limit, offset)

        return tuple(
            SearchResult(
                kind="message",
                id=str(row["id"]),
                conversation_id=str(row["conversation_id"]),
                snippet=_clean(str(row["excerpt"] or row["content"])),
                score=_fts_score(row["score"]),
                created_at=float(row["created_at"]),
                role=str(row["role"]),
            )
            for row in rows
        )

    def _search_messages_like(
        self,
        tokens: list[str],
        conversation_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[SearchResult, ...]:
        conditions = []
        params: list[Any] = []
        for token in tokens:
            conditions.append("content LIKE ? ESCAPE '\\'")
            params.append(_like_term(token))
        sql = (
            "SELECT id, conversation_id, role, content, created_at FROM messages WHERE "
            + " AND ".join(conditions)
        )
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY created_at DESC, id ASC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(max(0, offset))])

        rows = self._db.connection().execute(sql, params).fetchall()
        return tuple(
            SearchResult(
                kind="message",
                id=str(row["id"]),
                conversation_id=str(row["conversation_id"]),
                snippet=_make_snippet(str(row["content"]), tokens),
                score=1.0,
                created_at=float(row["created_at"]),
                role=str(row["role"]),
            )
            for row in rows
        )

    # -- memories ----------------------------------------------------------

    def search_memories(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[SearchResult, ...]:
        if limit <= 0:
            return ()
        tokens = _tokenize(query)
        if not tokens:
            return ()

        if self.backend == BACKEND_FTS5:
            return self._search_memories_fts(tokens, category, limit, offset)
        return self._search_memories_like(tokens, category, limit, offset)

    def _search_memories_fts(
        self,
        tokens: list[str],
        category: str | None,
        limit: int,
        offset: int,
    ) -> tuple[SearchResult, ...]:
        match = _fts_match_expression(tokens)
        sql = (
            "SELECT m.id AS id, m.content AS content, m.category AS category, "
            "m.created_at AS created_at, "
            "snippet(memories_fts, 0, '[', ']', '...', 12) AS excerpt, "
            "bm25(memories_fts) AS score "
            "FROM memories_fts JOIN memories m ON m.id = memories_fts.memory_id "
            "WHERE memories_fts MATCH ?"
        )
        params: list[Any] = [match]
        if category:
            sql += " AND m.category = ?"
            params.append(category)
        sql += " ORDER BY score ASC, m.created_at DESC, m.id ASC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(max(0, offset))])

        try:
            rows = self._db.connection().execute(sql, params).fetchall()
        except Exception as exc:  # FTS problem -> deterministic fallback
            logger.warning("FTS memory search failed (%s); using LIKE fallback", exc)
            return self._search_memories_like(tokens, category, limit, offset)

        return tuple(
            SearchResult(
                kind="memory",
                id=str(row["id"]),
                conversation_id=None,
                snippet=_clean(str(row["excerpt"] or row["content"])),
                score=_fts_score(row["score"]),
                created_at=float(row["created_at"]),
                category=str(row["category"]),
            )
            for row in rows
        )

    def _search_memories_like(
        self,
        tokens: list[str],
        category: str | None,
        limit: int,
        offset: int,
    ) -> tuple[SearchResult, ...]:
        conditions = []
        params: list[Any] = []
        for token in tokens:
            conditions.append("content LIKE ? ESCAPE '\\'")
            params.append(_like_term(token))
        sql = (
            "SELECT id, content, category, created_at FROM memories WHERE "
            + " AND ".join(conditions)
        )
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY created_at DESC, id ASC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(max(0, offset))])

        rows = self._db.connection().execute(sql, params).fetchall()
        return tuple(
            SearchResult(
                kind="memory",
                id=str(row["id"]),
                conversation_id=None,
                snippet=_make_snippet(str(row["content"]), tokens),
                score=1.0,
                created_at=float(row["created_at"]),
                category=str(row["category"]),
            )
            for row in rows
        )

    def search_all(self, query: str, *, limit: int = 20) -> tuple[SearchResult, ...]:
        """Search both stores; messages first, then memories."""
        if limit <= 0:
            return ()
        messages = self.search_messages(query, limit=limit)
        remaining = max(0, int(limit) - len(messages))
        memories = self.search_memories(query, limit=remaining) if remaining else ()
        return messages + memories


def _tokenize(query: Any) -> list[str]:
    """Split a query into word tokens; punctuation is dropped safely."""
    if not isinstance(query, str):
        return []
    return [token for token in _TOKEN_PATTERN.findall(query) if token][:16]


def _fts_match_expression(tokens: list[str]) -> str:
    """Build a safe FTS5 MATCH expression: quoted tokens joined by OR."""
    parts = []
    for token in tokens:
        escaped = token.replace('"', '""')
        parts.append(f'"{escaped}"')
    return " OR ".join(parts)


def _like_term(token: str) -> str:
    """Escape a token for `LIKE ... ESCAPE '\\'` (value is still bound)."""
    escaped = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _fts_score(raw: Any) -> float:
    """bm25() returns negative numbers; higher means a better match here."""
    try:
        return -float(raw)
    except (TypeError, ValueError):
        return 0.0


def _make_snippet(content: str, tokens: list[str]) -> str:
    lowered = content.lower()
    position = -1
    for token in tokens:
        position = lowered.find(token.lower())
        if position >= 0:
            break
    if position < 0:
        start = 0
    else:
        start = max(0, position - SNIPPET_CHARS // 3)
    excerpt = content[start : start + SNIPPET_CHARS]
    return _clean(excerpt)


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()[:SNIPPET_CHARS]
