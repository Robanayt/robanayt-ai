"""Database lifecycle: lazy creation, schema version, migrations, failures."""

from __future__ import annotations

import sqlite3

import pytest

import robanayt_ai.memory.database as database_module
from robanayt_ai.memory.database import (
    SCHEMA_VERSION,
    Database,
    create_legacy_database,
    map_sqlite_error,
)
from robanayt_ai.memory.errors import (
    DatabaseCorruptError,
    DatabaseLockedError,
    DatabaseUnavailableError,
    MigrationError,
)


def test_construction_touches_nothing(tmp_path) -> None:
    path = tmp_path / "runtime" / "memory" / "jarvis.db"

    db = Database(path)

    assert not path.exists()
    assert db.is_open is False
    assert db.exists is False


def test_open_creates_file_and_schema(tmp_path) -> None:
    db = Database(tmp_path / "sub" / "jarvis.db")

    connection = db.open()

    assert db.path.is_file()
    assert db.is_open is True
    assert db.user_version() == SCHEMA_VERSION
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
        )
    }
    assert {"conversations", "messages", "memories"} <= tables


def test_reopening_is_idempotent(tmp_path) -> None:
    path = tmp_path / "jarvis.db"
    with Database(path) as db:
        db.connection().execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, metadata) "
            "VALUES ('conv_1', 't', 1.0, 1.0, '{}')"
        )

    with Database(path) as db:
        assert db.user_version() == SCHEMA_VERSION
        count = db.connection().execute("SELECT count(*) FROM conversations").fetchone()[0]

    assert count == 1


def test_foreign_keys_are_enforced(tmp_path) -> None:
    with Database(tmp_path / "jarvis.db") as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.connection().execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at, seq) "
                "VALUES ('msg_1', 'missing', 'user', 'x', 1.0, 0)"
            )


def test_migration_from_version_one(tmp_path) -> None:
    path = create_legacy_database(tmp_path / "legacy.db", version=1)
    probe = sqlite3.connect(str(path))
    assert probe.execute("PRAGMA user_version").fetchone()[0] == 1
    probe.close()

    with Database(path) as db:
        assert db.user_version() == SCHEMA_VERSION
        columns = {
            row[1]
            for row in db.connection().execute("PRAGMA table_info(memories)")
        }

    assert {"source", "importance"} <= columns


def test_newer_schema_is_never_downgraded(tmp_path) -> None:
    path = tmp_path / "future.db"
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY)")
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    connection.commit()
    connection.close()

    with pytest.raises(MigrationError, match="newer than supported"):
        Database(path).open()

    # The file must still be there and untouched.
    assert path.is_file()
    check = sqlite3.connect(str(path))
    assert check.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION + 5
    check.close()


def test_corrupt_file_raises_and_is_left_untouched(tmp_path) -> None:
    path = tmp_path / "broken.db"
    path.write_bytes(b"this is definitely not a sqlite database" * 10)
    before = path.read_bytes()

    with pytest.raises(DatabaseCorruptError):
        Database(path).open()

    assert path.read_bytes() == before


def test_verify_integrity_detects_damage(tmp_path) -> None:
    path = tmp_path / "jarvis.db"
    with Database(path) as db:
        assert db.verify_integrity() is True

    with open(path, "r+b") as handle:
        handle.seek(4096)
        handle.write(b"\x00" * 512)

    with Database(path) as db:
        with pytest.raises(DatabaseCorruptError):
            db.verify_integrity()


def test_locked_database_is_reported(tmp_path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", boom)

    with pytest.raises(DatabaseLockedError):
        Database(tmp_path / "jarvis.db").open()


def test_io_failure_is_reported(tmp_path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(sqlite3, "connect", boom)

    with pytest.raises(DatabaseUnavailableError):
        Database(tmp_path / "jarvis.db").open()


def test_error_mapping_kinds() -> None:
    assert isinstance(
        map_sqlite_error(sqlite3.OperationalError("database is locked"), "p"),
        DatabaseLockedError,
    )
    assert isinstance(
        map_sqlite_error(sqlite3.DatabaseError("file is not a database"), "p"),
        DatabaseCorruptError,
    )
    assert isinstance(
        map_sqlite_error(sqlite3.OperationalError("no such table: x"), "p"),
        DatabaseUnavailableError,
    )


def test_transaction_rolls_back_on_error(tmp_path) -> None:
    db = Database(tmp_path / "jarvis.db")
    try:
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, metadata) "
                "VALUES ('conv_1', 't', 1.0, 1.0, '{}')"
            )
            raise RuntimeError("failure inside transaction")
    except RuntimeError:
        pass

    count = db.connection().execute("SELECT count(*) FROM conversations").fetchone()[0]
    assert count == 0
    db.close()


def test_transaction_commits(tmp_path) -> None:
    with Database(tmp_path / "jarvis.db") as db:
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, metadata) "
                "VALUES ('conv_1', 't', 1.0, 1.0, '{}')"
            )
        assert db.connection().execute("SELECT count(*) FROM conversations").fetchone()[0] == 1


def test_close_is_safe_twice(tmp_path) -> None:
    db = Database(tmp_path / "jarvis.db")
    db.open()
    db.close()
    db.close()
    assert db.is_open is False


def test_fts_detection_can_be_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(database_module, "supports_fts5", lambda connection: False)

    with Database(tmp_path / "jarvis.db") as db:
        assert db.fts_enabled is False
        tables = {
            row[0] for row in db.connection().execute("SELECT name FROM sqlite_master")
        }

    assert "messages_fts" not in tables


def test_fts5_is_available_in_this_runtime(tmp_path) -> None:
    with Database(tmp_path / "jarvis.db") as db:
        assert db.fts_enabled is True
        assert db.json_enabled is True
