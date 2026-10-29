"""SQLite storage for the memory layer.

Guarantees enforced here:

- The database file is created lazily: constructing `Database` touches nothing.
- Every statement with a variable value is parameterised - no string SQL.
- Schema changes are versioned and applied incrementally; a database that is
  NEWER than this code understands is never downgraded, rebuilt or deleted.
- A damaged file raises `DatabaseCorruptError` and is left untouched - the
  user decides what to do with it.
- No network access is ever performed.
- Threading: the connection is opened with `check_same_thread=False`, because
  a dialogue turn runs on a worker thread while the UI reads the history on
  its own. Access is serialized by one lock; multi-statement flows must go
  through `transaction()`. SQLite itself is used in its default serialized
  mode, so single statements from different threads cannot interleave into a
  corrupt state.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jarvis.memory.errors import (
    DatabaseCorruptError,
    DatabaseLockedError,
    DatabaseUnavailableError,
    MigrationError,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DEFAULT_TIMEOUT = 5.0

# --------------------------------------------------------------------------
# Schema definitions
# --------------------------------------------------------------------------

_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL,
        metadata    TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        created_at      REAL NOT NULL,
        seq             INTEGER NOT NULL DEFAULT 0,
        metadata        TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages (conversation_id, seq, created_at, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS memories (
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        category    TEXT NOT NULL DEFAULT 'general',
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL,
        source      TEXT,
        importance  INTEGER,
        tags        TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category)
    """,
)

# Full-text index (optional; created only when the runtime supports FTS5).
_FTS_TABLES = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content, message_id UNINDEXED, conversation_id UNINDEXED
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        content, memory_id UNINDEXED, category UNINDEXED
    )
    """,
)

# Incremental migrations: {from_version: (statement, ...)}
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        "ALTER TABLE memories ADD COLUMN source TEXT",
        "ALTER TABLE memories ADD COLUMN importance INTEGER",
    ),
}

# Version 1 shape, kept so migrations can be exercised (tests/recovery only).
_LEGACY_SCHEMAS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL,
            created_at REAL NOT NULL, updated_at REAL NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL,
            created_at REAL NOT NULL, seq INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            created_at REAL NOT NULL, updated_at REAL NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]'
        )
        """,
    ),
}


def create_legacy_database(path: str | Path, version: int = 1) -> Path:
    """Create a database at an older schema version (test/recovery helper)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    statements = _LEGACY_SCHEMAS.get(version)
    if statements is None:
        raise MigrationError(f"no legacy schema definition for version {version}")
    connection = sqlite3.connect(str(target))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in statements:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {int(version)}")
        connection.commit()
    finally:
        connection.close()
    return target


def supports_fts5(connection: sqlite3.Connection) -> bool:
    """Return True when this SQLite build can create FTS5 virtual tables."""
    try:
        connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        connection.execute("DROP TABLE _fts5_probe")
    except sqlite3.Error:
        return False
    return True


def supports_json1(connection: sqlite3.Connection) -> bool:
    """Return True when JSON1 table functions (`json_each`) are available."""
    try:
        connection.execute("SELECT count(*) FROM json_each('[]')").fetchone()
    except sqlite3.Error:
        return False
    return True


def map_sqlite_error(exc: sqlite3.Error, path: str | Path) -> Exception:
    """Translate a SQLite error into a typed memory-layer error."""
    message = str(exc).strip() or exc.__class__.__name__
    text = message.lower()
    if isinstance(exc, sqlite3.OperationalError):
        if "locked" in text or "busy" in text:
            return DatabaseLockedError(f"database is busy: {message} ({path})")
        if "corrupt" in text or "malformed" in text or "not a database" in text:
            return DatabaseCorruptError(
                f"database file is damaged: {message}; file left untouched ({path})"
            )
        return DatabaseUnavailableError(f"database unavailable: {message} ({path})")
    if isinstance(exc, sqlite3.DatabaseError):
        return DatabaseCorruptError(
            f"database file is damaged: {message}; file left untouched ({path})"
        )
    return DatabaseUnavailableError(f"database unavailable: {message} ({path})")


class Database:
    """Owns the SQLite connection, schema version and transactions."""

    def __init__(self, path: str | Path, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.path = Path(path)
        self.timeout = float(timeout)
        # One connection per thread: a dialogue turn runs on a worker thread
        # while the interface reads on its own. Sharing a single connection
        # between threads is what SQLite forbids.
        self._local = threading.local()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._generation = 0
        self._lock = threading.RLock()
        self._fts_enabled: bool | None = None
        self._json_enabled: bool | None = None

    def __repr__(self) -> str:
        state = "open" if self._connections else "closed"
        return f"Database(path={str(self.path)!r}, {state})"

    @property
    def is_open(self) -> bool:
        return bool(self._connections)

    @property
    def exists(self) -> bool:
        """Cheap filesystem check - does not open or create anything."""
        return self.path.is_file()

    @property
    def fts_enabled(self) -> bool:
        """True when the full-text index is available in this database."""
        if self._fts_enabled is None:
            self._fts_enabled = supports_fts5(self.connection())
        return self._fts_enabled

    @property
    def json_enabled(self) -> bool:
        if self._json_enabled is None:
            self._json_enabled = supports_json1(self.connection())
        return self._json_enabled

    def connection(self) -> sqlite3.Connection:
        """Return this thread's connection, opening the database on first use.

        Threads never share a connection; SQLite itself serializes concurrent
        access to the file (WAL, one writer at a time, `timeout` for busy
        waits).
        """
        cached = getattr(self._local, "entry", None)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        with self._lock:
            connection = self._connect()
            self._connections[threading.get_ident()] = connection
        self._local.entry = (self._generation, connection)
        return connection

    def open(self) -> sqlite3.Connection:
        """Open (or create) the database for the calling thread."""
        return self.connection()

    def _connect(self) -> sqlite3.Connection:
        """Create a connection and bring the schema up to date."""
        try:
            if self.path.parent and not self.path.parent.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                str(self.path),
                timeout=self.timeout,
                isolation_level=None,
                # Defensive: a connection stays with its thread, but a caller
                # that hands one over must not crash with a threading error.
                check_same_thread=False,
            )
        except OSError as exc:
            raise DatabaseUnavailableError(
                f"cannot create database directory/file {self.path}: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            raise map_sqlite_error(exc, self.path) from exc

        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            try:
                connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.Error:
                logger.debug("WAL unavailable for %s; continuing", self.path)
            self._ensure_schema(connection)
        except (DatabaseCorruptError, MigrationError):
            connection.close()
            raise
        except sqlite3.Error as exc:
            connection.close()
            raise map_sqlite_error(exc, self.path) from exc

        logger.debug("memory database ready: %s (thread=%s)", self.path, threading.get_ident())
        return connection

    def close(self) -> None:
        """Close every connection this Database opened, in every thread."""
        with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
            self._generation += 1
            self._fts_enabled = None
            self._json_enabled = None
        self._local.entry = None
        for connection in connections:
            try:
                connection.close()
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                logger.warning("error closing memory database: %s", exc)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a group of related writes atomically on this thread's connection."""
        connection = self.connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover - defensive
                logger.warning("rollback failed")
            raise
        connection.execute("COMMIT")

    def user_version(self) -> int:
        row = self.connection().execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def verify_integrity(self) -> bool:
        """Run `PRAGMA quick_check`; raises `DatabaseCorruptError` on damage."""
        connection = self.connection()
        try:
            row = connection.execute("PRAGMA quick_check").fetchone()
        except sqlite3.Error as exc:
            raise map_sqlite_error(exc, self.path) from exc
        result = str(row[0]) if row else ""
        if result.lower() != "ok":
            raise DatabaseCorruptError(
                f"integrity check failed: {result}; file left untouched ({self.path})"
            )
        return True

    def __enter__(self) -> Database:
        self.open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        except sqlite3.Error as exc:
            raise map_sqlite_error(exc, self.path) from exc

        if version > SCHEMA_VERSION:
            raise MigrationError(
                f"database schema version {version} is newer than supported "
                f"version {SCHEMA_VERSION}; file left untouched ({self.path})"
            )

        connection.execute("BEGIN IMMEDIATE")
        try:
            if version == 0:
                for statement in _TABLES:
                    connection.execute(statement)
                version = SCHEMA_VERSION
            else:
                while version < SCHEMA_VERSION:
                    statements = _MIGRATIONS.get(version)
                    if statements is None:
                        raise MigrationError(
                            f"no migration path from schema version {version}"
                        )
                    for statement in statements:
                        connection.execute(statement)
                    version += 1

            self._create_fts_tables(connection)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except (MigrationError, DatabaseCorruptError):
            connection.execute("ROLLBACK")
            raise
        except sqlite3.Error as exc:
            connection.execute("ROLLBACK")
            raise map_sqlite_error(exc, self.path) from exc

    def _create_fts_tables(self, connection: sqlite3.Connection) -> None:
        if not supports_fts5(connection):
            self._fts_enabled = False
            logger.debug("FTS5 unavailable for %s; using LIKE fallback", self.path)
            return
        try:
            for statement in _FTS_TABLES:
                connection.execute(statement)
            self._fts_enabled = True
        except sqlite3.Error:
            # FTS5 is optional: search falls back to parameterised LIKE.
            self._fts_enabled = False
            logger.debug("FTS5 unavailable for %s; using LIKE fallback", self.path)
