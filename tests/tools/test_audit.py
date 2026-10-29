"""Audit log: completeness, no secrets, optional file and bus sinks."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tools.helpers import FakeClock
from jarvis.core.bus import Event, EventBus
from jarvis.tools.audit import TOOL_BLOCKED, TOOL_EXECUTED, AuditLog, scrub
from jarvis.tools.models import AuditRecord


def record(**overrides: object) -> AuditRecord:
    payload = {
        "execution_id": "exe_1",
        "tool": "echo_text",
        "version": "1.0.0",
        "timestamp": 1.0,
        "status": "success",
        "permission": "safe",
        "reason": "ok",
        "arguments_digest": "abc",
        "argument_keys": ("text",),
        "duration_ms": 1.5,
        "caller": "cli",
    }
    payload.update(overrides)
    return AuditRecord(**payload)  # type: ignore[arg-type]


def test_record_is_stored_and_readable() -> None:
    audit = AuditLog(clock=FakeClock())

    audit.record(record())
    audit.record(record(execution_id="exe_2", status="denied"))

    assert len(audit) == 2
    assert audit.last().execution_id == "exe_2"
    assert audit.as_dicts()[0]["status"] == "success"


def test_disabled_log_stores_nothing() -> None:
    audit = AuditLog(clock=FakeClock(), enabled=False)
    audit.record(record())

    assert len(audit) == 0


def test_scrub_removes_secret_assignments() -> None:
    text = "failed: api_key=sk-verysecret password: hunter2 token=abc123"
    cleaned = scrub(text)

    assert "sk-verysecret" not in cleaned
    assert "hunter2" not in cleaned
    assert "abc123" not in cleaned
    assert "***" in cleaned


def test_scrub_leaves_ordinary_text_alone() -> None:
    assert scrub("connection refused by 127.0.0.1") == "connection refused by 127.0.0.1"
    assert scrub(None) is None


def test_error_is_scrubbed_before_storage() -> None:
    audit = AuditLog(clock=FakeClock())
    audit.record(record(status="failed", error="refused: token=supersecret"))

    assert "supersecret" not in audit.to_json()
    assert audit.last().error is not None and "***" in audit.last().error


def test_bus_receives_success_and_blocked_events() -> None:
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(TOOL_EXECUTED, seen.append)
    bus.subscribe(TOOL_BLOCKED, seen.append)

    audit = AuditLog(bus=bus, clock=FakeClock())
    audit.record(record(status="success"))
    audit.record(record(status="denied"))

    assert [event.type for event in seen] == [TOOL_EXECUTED, TOOL_BLOCKED]
    assert seen[0].payload["tool"] == "echo_text"


def test_file_sink_writes_one_json_line_per_record(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "tools.jsonl"
    audit = AuditLog(path=path, clock=FakeClock())

    audit.record(record())
    audit.record(record(execution_id="exe_2"))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"execution_id": "exe_2"' in lines[1]
    assert path.parent.is_dir()


def test_unwritable_file_does_not_break_the_caller(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    audit = AuditLog(path=blocker / "tools.jsonl", clock=FakeClock())

    audit.record(record())

    assert len(audit) == 1  # in-memory record survives the I/O failure


def test_max_records_limit_is_enforced() -> None:
    audit = AuditLog(clock=FakeClock(), max_records=2)
    for index in range(5):
        audit.record(record(execution_id=f"exe_{index}"))

    assert [item.execution_id for item in audit.records()] == ["exe_3", "exe_4"]


def test_clear_empties_the_log() -> None:
    audit = AuditLog(clock=FakeClock())
    audit.record(record())
    audit.clear()

    assert len(audit) == 0
    assert audit.last() is None


def test_audit_payload_has_no_unknown_fields() -> None:
    payload = record().as_dict()
    assert set(payload) == {
        "execution_id",
        "tool",
        "version",
        "timestamp",
        "status",
        "permission",
        "reason",
        "arguments_digest",
        "argument_keys",
        "duration_ms",
        "caller",
        "error",
    }
