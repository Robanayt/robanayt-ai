"""Conversation and message persistence.

Imported content is treated strictly as data: it is validated, stored as
strings and never executed, evaluated or interpreted.

Chronological ordering is stable even when timestamps are identical:
`(created_at, seq, id)` - `seq` is a per-conversation monotonic counter.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from robanayt_ai.memory.database import Database
from robanayt_ai.memory.errors import (
    ConversationNotFoundError,
    MemoryImportError,
    MemoryValidationError,
)
from robanayt_ai.memory.models import (
    Conversation,
    Message,
    decode_json,
    encode_json,
    new_id,
    validate_metadata,
    validate_text,
)

logger = logging.getLogger(__name__)

EXPORT_VERSION = 1

_ORDER_BY = "ORDER BY created_at ASC, seq ASC, id ASC"


class ConversationStore:
    """Persistence for conversations and their messages."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # -- conversations -----------------------------------------------------

    def create_conversation(
        self,
        title: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
        conversation_id: str | None = None,
        now: float | None = None,
    ) -> Conversation:
        timestamp = time.time() if now is None else float(now)
        conversation = Conversation(
            id=conversation_id or new_id("conv"),
            title=title or "",
            created_at=timestamp,
            updated_at=timestamp,
            metadata=validate_metadata(metadata),
        )
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    conversation.id,
                    conversation.title,
                    conversation.created_at,
                    conversation.updated_at,
                    encode_json(dict(conversation.metadata)),
                ),
            )
        logger.debug("conversation created: %s", conversation.id)
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = self._db.connection().execute(
            "SELECT id, title, created_at, updated_at, metadata FROM conversations "
            "WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return _row_to_conversation(row) if row is not None else None

    def require_conversation(self, conversation_id: str) -> Conversation:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(f"conversation not found: {conversation_id}")
        return conversation

    def list_conversations(
        self, *, limit: int = 50, offset: int = 0
    ) -> tuple[Conversation, ...]:
        if limit <= 0:
            return ()
        rows = self._db.connection().execute(
            "SELECT id, title, created_at, updated_at, metadata FROM conversations "
            "ORDER BY updated_at DESC, id ASC LIMIT ? OFFSET ?",
            (int(limit), int(max(0, offset))),
        ).fetchall()
        return tuple(_row_to_conversation(row) for row in rows)

    def count_conversations(self) -> int:
        row = self._db.connection().execute(
            "SELECT count(*) FROM conversations"
        ).fetchone()
        return int(row[0]) if row else 0

    def rename_conversation(self, conversation_id: str, title: str) -> Conversation:
        if not isinstance(title, str):
            raise MemoryValidationError("title must be a string")
        if len(title) > 512:
            raise MemoryValidationError("title is too long")
        conversation = self.require_conversation(conversation_id)
        updated = time.time()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, updated, conversation_id),
            )
        logger.debug("conversation renamed: %s", conversation_id)
        return Conversation(
            id=conversation.id,
            title=title,
            created_at=conversation.created_at,
            updated_at=updated,
            metadata=conversation.metadata,
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages explicitly."""
        if self.get_conversation(conversation_id) is None:
            return False
        with self._db.transaction() as conn:
            if self._db.fts_enabled:
                conn.execute(
                    "DELETE FROM messages_fts WHERE message_id IN "
                    "(SELECT id FROM messages WHERE conversation_id = ?)",
                    (conversation_id,),
                )
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        logger.info("conversation deleted: %s", conversation_id)
        return True

    # -- messages ----------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        created_at: float | None = None,
        message_id: str | None = None,
    ) -> Message:
        """Append one message; fails if the conversation does not exist."""
        self.require_conversation(conversation_id)
        timestamp = time.time() if created_at is None else float(created_at)
        message = Message(
            id=message_id or new_id("msg"),
            conversation_id=conversation_id,
            role=role,
            content=validate_text(content),
            created_at=timestamp,
            metadata=validate_metadata(metadata),
        )
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            seq = int(row[0]) + 1
            message = Message(
                id=message.id,
                conversation_id=message.conversation_id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                seq=seq,
                metadata=message.metadata,
            )
            conn.execute(
                "INSERT INTO messages "
                "(id, conversation_id, role, content, created_at, seq, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.created_at,
                    message.seq,
                    encode_json(dict(message.metadata)),
                ),
            )
            if self._db.fts_enabled:
                conn.execute(
                    "INSERT INTO messages_fts (content, message_id, conversation_id) "
                    "VALUES (?, ?, ?)",
                    (message.content, message.id, message.conversation_id),
                )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.created_at, conversation_id),
            )
        return message

    def get_messages(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[Message, ...]:
        """Messages in chronological order."""
        sql = (
            "SELECT id, conversation_id, role, content, created_at, seq, metadata "
            "FROM messages WHERE conversation_id = ? " + _ORDER_BY
        )
        params: list[Any] = [conversation_id]
        if limit is not None:
            if limit <= 0:
                return ()
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(max(0, offset))])
        rows = self._db.connection().execute(sql, params).fetchall()
        return tuple(_row_to_message(row) for row in rows)

    def recent_messages(self, conversation_id: str, count: int) -> tuple[Message, ...]:
        """The newest `count` messages, returned oldest-first."""
        if count <= 0:
            return ()
        rows = self._db.connection().execute(
            "SELECT id, conversation_id, role, content, created_at, seq, metadata "
            "FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at DESC, seq DESC, id DESC LIMIT ?",
            (conversation_id, int(count)),
        ).fetchall()
        return tuple(_row_to_message(row) for row in reversed(rows))

    def count_messages(self, conversation_id: str) -> int:
        row = self._db.connection().execute(
            "SELECT count(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    # -- export / import ---------------------------------------------------

    def export_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Return a JSON-safe export payload (data only, no code)."""
        conversation = self.require_conversation(conversation_id)
        messages = self.get_messages(conversation_id)
        return {
            "export_version": EXPORT_VERSION,
            "exported_at": time.time(),
            "conversation": conversation.to_dict(),
            "messages": [message.to_dict() for message in messages],
        }

    def import_conversation(self, payload: Mapping[str, Any]) -> Conversation:
        """Import a validated export; existing ids are never overwritten."""
        if not isinstance(payload, Mapping):
            raise MemoryImportError("import payload must be a JSON object")

        version = payload.get("export_version")
        if version != EXPORT_VERSION:
            raise MemoryImportError(
                f"unsupported export_version {version!r}; expected {EXPORT_VERSION}"
            )

        raw_conversation = payload.get("conversation")
        raw_messages = payload.get("messages")
        if not isinstance(raw_conversation, Mapping):
            raise MemoryImportError("import payload needs a 'conversation' object")
        if not isinstance(raw_messages, list):
            raise MemoryImportError("import payload needs a 'messages' list")

        conversation = _build_import_conversation(raw_conversation)
        messages = [_build_import_message(entry, conversation.id) for entry in raw_messages]

        # Duplicate ids must never replace existing data: remap instead.
        if self.get_conversation(conversation.id) is not None:
            new_conversation_id = new_id("conv")
            logger.info(
                "import: conversation id %s already exists; storing as %s",
                conversation.id,
                new_conversation_id,
            )
            conversation = Conversation(
                id=new_conversation_id,
                title=conversation.title,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                metadata=conversation.metadata,
            )
            messages = [
                Message(
                    id=message.id,
                    conversation_id=new_conversation_id,
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at,
                    seq=index,
                    metadata=message.metadata,
                )
                for index, message in enumerate(messages)
            ]

        existing_ids = self._existing_message_ids()
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    conversation.id,
                    conversation.title,
                    conversation.created_at,
                    conversation.updated_at,
                    encode_json(dict(conversation.metadata)),
                ),
            )
            for index, message in enumerate(messages):
                stored = message
                if stored.id in existing_ids:
                    stored = Message(
                        id=new_id("msg"),
                        conversation_id=stored.conversation_id,
                        role=stored.role,
                        content=stored.content,
                        created_at=stored.created_at,
                        seq=stored.seq,
                        metadata=stored.metadata,
                    )
                conn.execute(
                    "INSERT INTO messages "
                    "(id, conversation_id, role, content, created_at, seq, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        stored.id,
                        stored.conversation_id,
                        stored.role,
                        stored.content,
                        stored.created_at,
                        stored.seq if stored.seq else index,
                        encode_json(dict(stored.metadata)),
                    ),
                )
                if self._db.fts_enabled:
                    conn.execute(
                        "INSERT INTO messages_fts (content, message_id, conversation_id) "
                        "VALUES (?, ?, ?)",
                        (stored.content, stored.id, stored.conversation_id),
                    )
        return conversation

    def _existing_message_ids(self) -> set[str]:
        rows = self._db.connection().execute("SELECT id FROM messages").fetchall()
        return {str(row["id"]) for row in rows}


def _build_import_conversation(data: Mapping[str, Any]) -> Conversation:
    try:
        return Conversation.from_dict(data)
    except MemoryValidationError as exc:
        raise MemoryImportError(f"invalid conversation in import: {exc}") from exc


def _build_import_message(data: Any, conversation_id: str) -> Message:
    if not isinstance(data, Mapping):
        raise MemoryImportError("each message entry must be a JSON object")
    payload = dict(data)
    payload["conversation_id"] = conversation_id
    payload.setdefault("id", new_id("msg"))
    try:
        return Message.from_dict(payload)
    except MemoryValidationError as exc:
        raise MemoryImportError(f"invalid message in import: {exc}") from exc


def _row_to_conversation(row: Any) -> Conversation:
    return Conversation(
        id=str(row["id"]),
        title=str(row["title"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        metadata=decode_json(row["metadata"], {}),
    )


def _row_to_message(row: Any) -> Message:
    return Message(
        id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=float(row["created_at"]),
        seq=int(row["seq"]),
        metadata=decode_json(row["metadata"], {}),
    )


def message_sequence(values: Sequence[Message]) -> tuple[Message, ...]:
    """Utility: reorder messages deterministically (used by tests/tools)."""
    return tuple(sorted(values, key=lambda item: (item.created_at, item.seq, item.id)))
