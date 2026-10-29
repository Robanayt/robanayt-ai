"""View models: what the UI is allowed to see."""

from __future__ import annotations

import json

from jarvis.orchestration.models import (
    DialogResult,
    DialogStatus,
    PendingApproval,
    ToolCallRecord,
)
from jarvis.surfaces.protocol import (
    STATE_ERROR,
    STATE_IDLE,
    STATE_LABELS,
    STATE_THINKING,
    STATE_TOOL,
    STATE_WAITING,
    ChatMessage,
    pending_view,
    state_view,
    tool_call_view,
    turn_view,
)


def test_turn_view_keeps_status_and_tools() -> None:
    record = ToolCallRecord(
        tool="add_numbers", status="success", sequence=1, execution_id="exec_1"
    )
    result = DialogResult(
        status=DialogStatus.SUCCESS,
        answer="42",
        conversation_id="conv_1",
        tool_calls=(record,),
        provider="ollama",
        model="llama3",
    )

    view = turn_view(result)

    assert view.ok is True
    assert view.status == "success"
    assert view.answer == "42"
    assert view.tool_calls[0].tool == "add_numbers"
    assert view.provider == "ollama"


def test_unsuccessful_status_is_not_ok() -> None:
    view = turn_view(DialogResult(status=DialogStatus.AI_ERROR, error="no provider"))

    assert view.ok is False
    assert view.error == "no provider"


def test_pending_view_shows_arguments_for_consent() -> None:
    approval = PendingApproval(
        pending_id="dlg_9",
        tool="write_file",
        version="2.0.0",
        arguments={"path": "a.txt", "text": "hello"},
        arguments_digest="digest",
    )

    view = pending_view(approval)

    assert view.arguments == {"path": "a.txt", "text": "hello"}
    assert json.loads(view.arguments_text())["path"] == "a.txt"


def test_state_view_has_a_human_label() -> None:
    view = state_view(state=STATE_WAITING)

    assert view.state_label == "ожидание подтверждения"


def test_every_state_has_a_label() -> None:
    for state in (
        STATE_IDLE,
        STATE_THINKING,
        "answering",
        STATE_TOOL,
        STATE_WAITING,
        STATE_ERROR,
        "offline",
    ):
        assert state in STATE_LABELS


def test_tool_call_view_carries_the_reason() -> None:
    view = tool_call_view(
        ToolCallRecord(tool="forbidden", status="blocked", reason="policy")
    )

    assert (view.tool, view.status, view.reason) == ("forbidden", "blocked", "policy")


def test_messages_are_plain_data() -> None:
    message = ChatMessage(role="user", text="привет", meta="")

    assert (message.role, message.text) == ("user", "привет")
