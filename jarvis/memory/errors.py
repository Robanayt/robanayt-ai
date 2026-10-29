"""Memory layer errors.

The base class is deliberately NOT called `MemoryError`: that name would
shadow the Python builtin and silently break unrelated `except` clauses.

Error mapping rules for the storage layer:

- `DatabaseUnavailableError` - the file cannot be opened/created (I/O).
- `DatabaseLockedError` - SQLite reports the database is locked or busy.
- `DatabaseCorruptError` - SQLite reports the file is not a valid database;
  the file is never deleted or rebuilt automatically.
- `MigrationError` - the on-disk schema is newer than this code supports;
  the database is left untouched.
"""

from __future__ import annotations


class MemoryLayerError(Exception):
    """Base class for every memory-layer failure."""


class MemoryDisabledError(MemoryLayerError):
    """Memory is disabled by configuration; no database is used."""


class MemoryValidationError(MemoryLayerError):
    """Data failed validation before reaching the database."""


class MemoryStoreError(MemoryLayerError):
    """Base class for persistence failures."""


class DatabaseUnavailableError(MemoryStoreError):
    """The database file cannot be opened, created or read."""


class DatabaseLockedError(MemoryStoreError):
    """SQLite reported the database as locked or busy."""


class DatabaseCorruptError(MemoryStoreError):
    """SQLite reported the file as damaged; it was not modified."""


class MigrationError(MemoryStoreError):
    """Schema migration is required but cannot be performed safely."""


class ConversationNotFoundError(MemoryLayerError):
    """The requested conversation does not exist."""


class SavedMemoryNotFoundError(MemoryLayerError):
    """The requested saved memory does not exist."""


class MemoryImportError(MemoryValidationError):
    """An import payload was rejected; nothing was written."""


class MemoryExportError(MemoryLayerError):
    """An export could not be written (bad path, unsafe filename, I/O)."""
