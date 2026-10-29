"""JARVIS_AI memory layer (Phase 3).

Local-first persistent memory: conversations, messages and explicitly saved
memories in a SQLite database. No network, no embeddings, no vector search,
no automatic extraction - memory is created only when the caller asks.

The database is opened lazily: importing this package or starting JARVIS
never creates or migrates anything.
"""

from robanayt_ai.memory.conversation_store import ConversationStore
from robanayt_ai.memory.database import SCHEMA_VERSION, Database
from robanayt_ai.memory.errors import (
    ConversationNotFoundError,
    DatabaseCorruptError,
    DatabaseLockedError,
    DatabaseUnavailableError,
    MemoryDisabledError,
    MemoryExportError,
    MemoryImportError,
    MemoryLayerError,
    MemoryStoreError,
    MemoryValidationError,
    MigrationError,
    SavedMemoryNotFoundError,
)
from robanayt_ai.memory.memory_store import MemoryStore
from robanayt_ai.memory.models import (
    Conversation,
    MemoryStats,
    Message,
    PreparedContext,
    SavedMemory,
    SearchResult,
)
from robanayt_ai.memory.search import SearchEngine
from robanayt_ai.memory.service import MemoryService, TurnResult

__all__ = [
    "Conversation",
    "ConversationNotFoundError",
    "ConversationStore",
    "Database",
    "DatabaseCorruptError",
    "DatabaseLockedError",
    "DatabaseUnavailableError",
    "MemoryDisabledError",
    "MemoryExportError",
    "MemoryImportError",
    "MemoryLayerError",
    "MemoryService",
    "MemoryStats",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryValidationError",
    "Message",
    "MigrationError",
    "PreparedContext",
    "SavedMemory",
    "SavedMemoryNotFoundError",
    "SCHEMA_VERSION",
    "SearchEngine",
    "SearchResult",
    "TurnResult",
]
