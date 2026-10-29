"""Dialog orchestration data models (Phase 5).

The result of one user message is always a `DialogResult`: it carries the
final answer only when `status` is `success`, and every other outcome is
reported under its own status instead of being hidden behind an empty answer.

Nothing here stores a traceback, an API key or a full provider payload. Tool
arguments are kept in a pending approval (so the user can give informed
consent) but are excluded from the safe `describe()`/`as_dict()` view and
never written to a log.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_MESSAGE_CHARS = 8_000
MAX_TOOL_RESULT_CHARS = 2_000
MAX_PENDING_TURNS = 8
DEFAULT_MEMORY_SNIPPETS = 5


class DialogStatus(str, Enum):
    """Outcome of one user message.

    Only `SUCCESS` means: the dialogue turn finished and `answer` is the
    assistant's reply. Everything else explains why it did not.
    """

    SUCCESS = "success"
    APPROVAL_REQUIRED = "approval_required"
    TOOL_LIMIT_REACHED = "tool_limit_reached"
    INVALID_TOOL_CALL = "invalid_tool_call"
    AI_ERROR = "ai_error"
    MEMORY_ERROR = "memory_error"
    TOOL_ERROR = "tool_error"
    EMPTY_RESPONSE = "empty_response"
    PENDING_EXPIRED = "pending_expired"
    DISABLED = "disabled"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ToolCallRecord:
    """One executed (or refused) tool call inside a turn."""

    tool: str
    status: str
    sequence: int = 0
    execution_id: str = ""
    reason: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "sequence": self.sequence,
            "execution_id": self.execution_id,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass(frozen=True)
class PendingApproval:
    """A tool call that stopped the turn and waits for a trusted decision.

    `arguments` is kept so the user can see exactly what would happen; the
    digest is what an approval token is bound to.
    """

    pending_id: str
    tool: str
    version: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    arguments_digest: str = ""
    reason: str = ""
    conversation_id: str | None = None
    created_at: float = 0.0
    expires_at: float = 0.0

    def is_expired(self, now: float) -> bool:
        return now > self.expires_at

    def describe(self) -> dict[str, Any]:
        """Safe view: no argument values, no private text."""
        return {
            "pending_id": self.pending_id,
            "tool": self.tool,
            "version": self.version,
            "arguments_digest": self.arguments_digest,
            "argument_keys": sorted(str(key) for key in self.arguments),
            "reason": self.reason,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    def as_dict(self, *, include_arguments: bool = False) -> dict[str, Any]:
        payload = self.describe()
        if include_arguments:
            payload["arguments"] = dict(self.arguments)
        return payload


@dataclass(frozen=True)
class DialogResult:
    """Structured outcome of one user message."""

    status: DialogStatus
    answer: str = ""
    conversation_id: str | None = None
    tool_calls: tuple[ToolCallRecord, ...] = ()
    pending: PendingApproval | None = None
    error: str | None = None
    error_type: str | None = None
    persisted: bool = False
    persist_error: str | None = None
    memory_enabled: bool = True
    context_truncated: bool = False
    provider: str | None = None
    model: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is DialogStatus.SUCCESS

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def as_dict(self, *, include_pending_arguments: bool = False) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "ok": self.ok,
            "answer": self.answer,
            "conversation_id": self.conversation_id,
            "tool_calls": [record.as_dict() for record in self.tool_calls],
            "tool_call_count": self.tool_call_count,
            "pending": (
                self.pending.as_dict(include_arguments=include_pending_arguments)
                if self.pending
                else None
            ),
            "error": self.error,
            "error_type": self.error_type,
            "persisted": self.persisted,
            "persist_error": self.persist_error,
            "memory_enabled": self.memory_enabled,
            "context_truncated": self.context_truncated,
            "provider": self.provider,
            "model": self.model,
        }
