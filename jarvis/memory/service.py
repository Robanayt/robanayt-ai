"""MemoryService - orchestration entry point for the memory layer.

Rules:

- Constructing the service performs no I/O; the database is opened on the
  first real operation (lazy initialisation).
- If memory is disabled, every operation raises `MemoryDisabledError` with a
  clear message instead of silently doing nothing.
- Log records contain identifiers and counts only - never message content,
  never saved-memory text.
- AI orchestration never happens implicitly: `run_turn()` is called
  deliberately by the caller with an explicitly injected AI service.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from jarvis.core.config import MemoryConfig
from jarvis.memory.conversation_store import ConversationStore
from jarvis.memory.database import Database
from jarvis.memory.errors import (
    MemoryDisabledError,
    MemoryExportError,
    MemoryImportError,
    MemoryLayerError,
)
from jarvis.memory.memory_store import MemoryStore
from jarvis.memory.models import (
    Conversation,
    MemoryStats,
    Message,
    PreparedContext,
    SavedMemory,
    SearchResult,
    new_id,
    validate_text,
)
from jarvis.memory.search import SearchEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one explicit conversation turn."""

    conversation_id: str
    user_message: Message | None
    assistant_message: Message | None
    response: Any
    persisted: bool


class MemoryService:
    """High-level memory facade used by the rest of JARVIS."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        database: Database | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self._database = database
        self._conversations: ConversationStore | None = None
        self._memories: MemoryStore | None = None
        self._search: SearchEngine | None = None

    def __repr__(self) -> str:
        state = "disabled" if not self.config.enabled else "enabled"
        return f"MemoryService({state}, path={str(self.database_path)!r})"

    # -- lifecycle ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def database_path(self) -> Path:
        """Configured database path. Pure computation - no I/O."""
        return self.config.database_path

    @property
    def is_initialized(self) -> bool:
        """True when the database file already exists on disk."""
        if self._database is not None and self._database.is_open:
            return True
        return self.database_path.is_file()

    @property
    def is_open(self) -> bool:
        return self._database is not None and self._database.is_open

    def close(self) -> None:
        if self._database is not None:
            self._database.close()
        self._conversations = None
        self._memories = None
        self._search = None

    def __enter__(self) -> MemoryService:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _require_database(self) -> Database:
        if not self.config.enabled:
            raise MemoryDisabledError(
                "memory is disabled by configuration (memory.enabled=false)"
            )
        if self._database is None:
            self._database = Database(
                self.database_path, timeout=self.config.database_timeout
            )
        return self._database

    @property
    def conversations(self) -> ConversationStore:
        database = self._require_database()
        if self._conversations is None:
            self._conversations = ConversationStore(database)
        return self._conversations

    @property
    def memories(self) -> MemoryStore:
        database = self._require_database()
        if self._memories is None:
            self._memories = MemoryStore(database)
        return self._memories

    @property
    def search(self) -> SearchEngine:
        database = self._require_database()
        if self._search is None:
            self._search = SearchEngine(database)
        return self._search

    # -- diagnostics -------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Diagnostics without private data and without opening the database."""
        return {
            "enabled": self.config.enabled,
            "database_path": str(self.database_path),
            "initialized": self.is_initialized,
            "open": self.is_open,
            "max_recent_messages": self.config.max_recent_messages,
            "default_search_limit": self.config.default_search_limit,
            "max_context_chars": self.config.max_context_chars,
        }

    def stats(self) -> MemoryStats:
        """Counts and schema information. Opens the database if needed."""
        database = self._require_database()
        connection = database.connection()
        conversations = int(connection.execute("SELECT count(*) FROM conversations").fetchone()[0])
        messages = int(connection.execute("SELECT count(*) FROM messages").fetchone()[0])
        memories = int(connection.execute("SELECT count(*) FROM memories").fetchone()[0])
        size = database.path.stat().st_size if database.path.is_file() else None
        return MemoryStats(
            conversations=conversations,
            messages=messages,
            memories=memories,
            schema_version=database.user_version(),
            database_path=str(database.path),
            database_bytes=size,
            search_backend=self.search.backend,
        )

    # -- context preparation ----------------------------------------------

    def prepare_context(
        self,
        conversation_id: str | None,
        *,
        system_prompt: str | None = None,
        max_messages: int | None = None,
        max_chars: int | None = None,
        memory_snippets: Sequence[str] = (),
    ) -> PreparedContext:
        """Build AI-ready context from stored messages.

        Limits: newest messages win. The newest message is always kept even
        if it alone exceeds the character budget. `chars` counts characters,
        not tokens.
        """
        limit = int(max_messages or self.config.max_recent_messages)
        budget = self.config.max_context_chars if max_chars is None else max_chars

        if conversation_id is None:
            return PreparedContext(
                system_prompt=system_prompt,
                messages=(),
                memory_snippets=tuple(memory_snippets),
                truncated=False,
                conversation_id=None,
            )

        messages = self.conversations.recent_messages(conversation_id, limit)
        selected: list[Message] = []
        truncated = len(messages) == limit and self.conversations.count_messages(
            conversation_id
        ) > limit

        if budget and budget > 0:
            used = 0
            for message in reversed(messages):
                cost = len(message.content)
                if selected and used + cost > budget:
                    truncated = True
                    continue
                selected.append(message)
                used += cost
            selected.reverse()
        else:
            selected = list(messages)

        return PreparedContext(
            system_prompt=system_prompt,
            messages=tuple(selected),
            memory_snippets=tuple(memory_snippets),
            truncated=truncated,
            conversation_id=conversation_id,
        )

    def select_memory_snippets(
        self,
        query: str,
        *,
        limit: int | None = None,
        category: str | None = None,
    ) -> tuple[str, ...]:
        """Explicitly selected memory text for context (never automatic)."""
        results = self.search.search_memories(
            query, category=category, limit=int(limit or self.config.default_search_limit)
        )
        return tuple(result.snippet for result in results)

    # -- AI orchestration --------------------------------------------------

    def run_turn(
        self,
        ai_service: Any,
        conversation_id: str,
        prompt: str,
        *,
        persist: bool = True,
        system_prompt: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_messages: int | None = None,
        memory_snippets: Sequence[str] = (),
    ) -> TurnResult:
        """One explicit conversation turn: persist -> context -> AI -> persist.

        The AI service is injected by the caller, so this method never starts
        a request on its own and never contacts anything implicitly.

        The prompt always reaches the model. When `persist=False` it is added
        to the request as a transient message and nothing is written to the
        database.
        """
        if not self.config.enabled:
            raise MemoryDisabledError("memory is disabled by configuration")

        prompt = validate_text(prompt, "prompt")
        conversation = self.conversations.require_conversation(conversation_id)
        user_message: Message | None = None
        if persist:
            user_message = self.conversations.add_message(conversation_id, "user", prompt)

        context = self.prepare_context(
            conversation_id,
            system_prompt=system_prompt,
            max_messages=max_messages,
            memory_snippets=memory_snippets,
        )
        if not persist:
            transient = Message(
                id=new_id("msg"),
                conversation_id=conversation.id,
                role="user",
                content=prompt,
                created_at=time.time(),
            )
            context = replace(context, messages=context.messages + (transient,))

        request = context.to_ai_request(model=model)
        response = ai_service.generate(request, provider=provider)

        assistant_message: Message | None = None
        content = getattr(response, "content", "") or ""
        if persist:
            if content.strip():
                assistant_message = self.conversations.add_message(
                    conversation_id, "assistant", content
                )
            else:
                logger.warning(
                    "assistant response was empty; nothing persisted (conversation=%s)",
                    conversation.id,
                )

        logger.info(
            "conversation turn complete (conversation=%s, persisted=%s)",
            conversation.id,
            persist,
        )
        return TurnResult(
            conversation_id=conversation.id,
            user_message=user_message,
            assistant_message=assistant_message,
            response=response,
            persisted=persist,
        )

    # -- export / import ---------------------------------------------------

    def export_conversation_to_file(
        self,
        conversation_id: str,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Write a conversation export to a safe, validated path."""
        target = safe_output_path(path)
        if target.exists() and not overwrite:
            raise MemoryExportError(
                f"refusing to overwrite existing file: {target} (use overwrite=True)"
            )
        payload = self.conversations.export_conversation(conversation_id)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise MemoryExportError(f"cannot write export to {target}: {exc}") from exc
        logger.info("conversation exported (file=%s)", target.name)
        return target

    def import_conversation_from_file(self, path: str | Path) -> Conversation:
        """Import a validated conversation export (JSON only, never executed)."""
        source = Path(path)
        if not source.is_file():
            raise MemoryImportError(f"import file not found: {source}")
        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise MemoryImportError(f"cannot read import file {source}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise MemoryImportError(f"invalid JSON in {source}: {exc}") from exc
        return self.conversations.import_conversation(payload)

    # -- convenience wrappers ---------------------------------------------

    def create_conversation(self, title: str = "", **kwargs: Any) -> Conversation:
        return self.conversations.create_conversation(title, **kwargs)

    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        return self.conversations.add_message(conversation_id, role, content)

    def remember(
        self,
        content: str,
        *,
        category: str = "general",
        tags: Sequence[str] = (),
        importance: int | None = None,
        source: str | None = None,
    ) -> SavedMemory:
        """Explicitly save one long-term memory."""
        return self.memories.save(
            content, category=category, tags=tags, importance=importance, source=source
        )

    def recall(self, query: str, *, limit: int | None = None) -> tuple[SearchResult, ...]:
        limit_value = int(limit or self.config.default_search_limit)
        return self.search.search_all(query, limit=limit_value)


def safe_output_path(path: str | Path) -> Path:
    """Validate an application-controlled output path (no traversal)."""
    target = Path(path).expanduser()
    parts = target.parts
    if ".." in parts:
        raise MemoryExportError(f"path must not contain '..': {path}")
    if target.is_absolute() and target.drive and len(parts) <= 1:
        raise MemoryExportError(f"refusing to write to drive root: {path}")
    return target


def utc_now() -> float:
    """Current timestamp (epoch seconds). Exposed for deterministic callers."""
    return time.time()


def is_memory_error(exc: BaseException) -> bool:
    """True when an exception belongs to the memory layer."""
    return isinstance(exc, MemoryLayerError)


def build_memory_service(config: MemoryConfig | None = None) -> MemoryService:
    """Compose a memory service. Performs no I/O and no network access."""
    return MemoryService(config)


def summarise_results(results: Sequence[SearchResult]) -> tuple[Mapping[str, Any], ...]:
    """Short, privacy-aware summaries for console output."""
    return tuple(
        {
            "kind": result.kind,
            "id": result.id,
            "conversation_id": result.conversation_id,
            "created_at": result.created_at,
            "snippet_chars": len(result.snippet),
        }
        for result in results
    )
