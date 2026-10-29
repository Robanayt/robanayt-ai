"""View models for the desktop UI.

The UI never touches `DialogResult` or the tool registry directly: the bridge
converts core objects into these plain structures. They carry only what is
safe to display - statuses, names, counters and (for a pending call) the
arguments the user must see before deciding. No credentials, no traces, no
raw provider payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from jarvis.orchestration.models import (
    DialogResult,
    PendingApproval,
    ToolCallRecord,
)

# Application states the UI may show. Every state is derived from real work:
# an idle process shows "idle", never a decorative spinner.
STATE_IDLE = "idle"
STATE_THINKING = "thinking"
STATE_ANSWERING = "answering"
STATE_TOOL = "tool"
STATE_WAITING = "waiting"
STATE_ERROR = "error"
STATE_OFFLINE = "offline"
STATE_CANCELLING = "cancelling"
STATE_CANCELLED = "cancelled"

STATE_LABELS: dict[str, str] = {
    STATE_IDLE: "ожидание",
    STATE_THINKING: "обработка запроса",
    STATE_ANSWERING: "формирование ответа",
    STATE_TOOL: "выполнение инструмента",
    STATE_WAITING: "ожидание подтверждения",
    STATE_ERROR: "ошибка",
    STATE_OFFLINE: "провайдер недоступен",
    STATE_CANCELLING: "отмена хода",
    STATE_CANCELLED: "ход отменён",
}

# States in which the core is really doing work. Cancelling makes sense only
# here; an idle or waiting process reports itself as such.
BUSY_STATES: frozenset[str] = frozenset(
    {STATE_THINKING, STATE_ANSWERING, STATE_TOOL, STATE_CANCELLING}
)


@dataclass(frozen=True)
class ChatMessage:
    """One line in the transcript."""

    role: str  # "user" | "assistant" | "system"
    text: str
    meta: str = ""


@dataclass(frozen=True)
class ToolCallView:
    tool: str
    status: str
    reason: str = ""
    execution_id: str = ""


@dataclass(frozen=True)
class PendingView:
    """A call that stopped the turn and waits for the user."""

    pending_id: str
    tool: str
    version: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    digest: str = ""
    expires_at: float = 0.0

    def arguments_text(self) -> str:
        """Arguments as JSON for the confirmation card."""
        import json

        try:
            return json.dumps(dict(self.arguments), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(self.arguments)


@dataclass(frozen=True)
class VoiceView:
    """Voice state shown next to the dialogue state - never decorative."""

    state: str
    label: str
    available: bool = False
    stt_provider: str | None = None
    tts_provider: str | None = None
    detail: str = ""

    @property
    def active(self) -> bool:
        return self.state in {
            "listening",
            "transcribing",
            "thinking",
            "speaking",
            "cancelling",
        }


@dataclass(frozen=True)
class TurnView:
    status: str
    answer: str = ""
    conversation_id: str | None = None
    tool_calls: tuple[ToolCallView, ...] = ()
    pending: PendingView | None = None
    error: str | None = None
    error_type: str | None = None
    provider: str | None = None
    model: str | None = None
    persisted: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass(frozen=True)
class StateView:
    """Everything the UI needs to draw one frame."""

    state: str
    conversation_id: str | None = None
    messages: tuple[ChatMessage, ...] = ()
    pending: tuple[PendingView, ...] = ()
    providers: tuple[str, ...] = ()
    mode: str = "normal"
    tools: tuple[tuple[str, str], ...] = ()
    limits: Mapping[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    voice: VoiceView | None = None

    @property
    def state_label(self) -> str:
        return STATE_LABELS.get(self.state, self.state)


def voice_view(
    *,
    state: str,
    label: str,
    available: bool = False,
    stt_provider: str | None = None,
    tts_provider: str | None = None,
    detail: str = "",
) -> VoiceView:
    return VoiceView(
        state=state,
        label=label,
        available=available,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        detail=detail,
    )


def tool_call_view(record: ToolCallRecord) -> ToolCallView:
    return ToolCallView(
        tool=record.tool,
        status=record.status,
        reason=record.reason,
        execution_id=record.execution_id,
    )


def pending_view(approval: PendingApproval) -> PendingView:
    return PendingView(
        pending_id=approval.pending_id,
        tool=approval.tool,
        version=approval.version,
        arguments=dict(approval.arguments),
        reason=approval.reason,
        digest=approval.arguments_digest,
        expires_at=approval.expires_at,
    )


def turn_view(result: DialogResult) -> TurnView:
    status = getattr(result.status, "value", str(result.status))
    return TurnView(
        status=status,
        answer=result.answer or "",
        conversation_id=result.conversation_id,
        tool_calls=tuple(tool_call_view(record) for record in result.tool_calls),
        pending=pending_view(result.pending) if result.pending is not None else None,
        error=result.error,
        error_type=result.error_type,
        provider=result.provider,
        model=result.model,
        persisted=result.persisted,
    )


def state_view(
    *,
    state: str,
    conversation_id: str | None = None,
    messages: tuple[ChatMessage, ...] = (),
    pending: tuple[PendingView, ...] = (),
    providers: tuple[str, ...] = (),
    mode: str = "normal",
    tools: tuple[tuple[str, str], ...] = (),
    limits: Mapping[str, Any] | None = None,
    last_error: str | None = None,
    voice: VoiceView | None = None,
) -> StateView:
    return StateView(
        state=state,
        conversation_id=conversation_id,
        messages=messages,
        pending=pending,
        providers=providers,
        mode=mode,
        tools=tools,
        limits=dict(limits or {}),
        last_error=last_error,
        voice=voice,
    )
