"""Execution audit (Phase 4).

What is stored: tool identity, execution id, time, final status, the
permission decision, a digest of the arguments and a scrubbed error.

What is never stored: argument values, API keys, tokens, passwords or the
full text of user messages. Errors are passed through `scrub()` first, and
the argument digest only allows correlation with an approval - it cannot be
reversed into the original data.

The log can publish to the core `EventBus`, which makes it usable by the
core logging/journal integration later without changing this module.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from robanayt_ai.core.bus import Event, EventBus
from robanayt_ai.tools.models import AuditRecord

logger = logging.getLogger(__name__)

TOOL_EXECUTED = "tool.executed"
TOOL_BLOCKED = "tool.blocked"

# Matches `password: hunter2`, `api_key=abc`, `token: ...` and similar.
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([a-z0-9_.-]*(?:api[_\-]?key|token|secret|password|passwd|pwd|"
    r"authorization|bearer|credential)[a-z0-9_.-]*)\s*[:=]\s*(\S+)"
)


def scrub(text: str | None) -> str | None:
    """Replace obvious secret assignments with a placeholder."""
    if not text:
        return text
    return _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=***", text)


class AuditLog:
    """Append-only, in-memory audit with optional bus and file sinks."""

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        path: Path | str | None = None,
        clock: Callable[[], float] = time.time,
        max_records: int = 1_000,
        enabled: bool = True,
    ) -> None:
        self._records: list[AuditRecord] = []
        self._bus = bus
        self._path = Path(path) if path is not None else None
        self._clock = clock
        self._max_records = max(1, int(max_records))
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, record: AuditRecord) -> AuditRecord:
        """Store one attempt. Never raises into the caller's happy path."""
        if not self._enabled:
            return record
        safe = AuditRecord(
            **{**record.as_dict(), "error": scrub(record.error), "argument_keys": record.argument_keys}
        )
        self._records.append(safe)
        if len(self._records) > self._max_records:
            del self._records[: len(self._records) - self._max_records]
        self._publish(safe)
        self._write(safe)
        return safe

    def _publish(self, record: AuditRecord) -> None:
        if self._bus is None:
            return
        event_type = TOOL_EXECUTED if record.status == "success" else TOOL_BLOCKED
        try:
            self._bus.publish(Event(event_type, record.as_dict()))
        except Exception:  # noqa: BLE001 - the bus records its own failures
            logger.warning("audit event could not be published: %s", record.execution_id)

    def _write(self, record: AuditRecord) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json() + "\n")
        except OSError as exc:
            logger.warning("audit file write failed (%s): %s", self._path, exc)

    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    def last(self) -> AuditRecord | None:
        return self._records[-1] if self._records else None

    def as_dicts(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self._records]

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def to_json(self) -> str:
        return json.dumps(self.as_dicts(), ensure_ascii=False, indent=2)
