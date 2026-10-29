"""Dialog models: statuses, safe views and structured results."""

from __future__ import annotations

from tests.orchestration.dialog_fakes import FakeClock
from jarvis.orchestration.models import (
    DialogResult,
    DialogStatus,
    PendingApproval,
    ToolCallRecord,
)


def test_only_success_is_ok() -> None:
    assert DialogResult(status=DialogStatus.SUCCESS).ok is True
    for status in DialogStatus:
        if status is not DialogStatus.SUCCESS:
            assert DialogResult(status=status).ok is False


def test_tool_call_count_comes_from_records() -> None:
    result = DialogResult(
        status=DialogStatus.SUCCESS,
        tool_calls=(
            ToolCallRecord(tool="a", status="success", sequence=1),
            ToolCallRecord(tool="b", status="failed", sequence=2, error="boom"),
        ),
    )
    assert result.tool_call_count == 2
    assert result.as_dict()["tool_call_count"] == 2


def test_pending_approval_hides_argument_values_by_default() -> None:
    pending = PendingApproval(
        pending_id="dlg_1",
        tool="echo_text",
        version="1.0.0",
        arguments={"text": "private note"},
        arguments_digest="abc",
        created_at=1.0,
        expires_at=61.0,
    )
    safe = pending.describe()

    assert "private note" not in str(safe)
    assert safe["argument_keys"] == ["text"]
    assert pending.as_dict() == safe
    assert pending.as_dict(include_arguments=True)["arguments"] == {"text": "private note"}


def test_pending_approval_expiry() -> None:
    clock = FakeClock(10.0)
    pending = PendingApproval(
        pending_id="dlg_1", tool="t", version="1", created_at=10.0, expires_at=20.0
    )
    assert pending.is_expired(clock()) is False
    clock.advance(11)
    assert pending.is_expired(clock()) is True


def test_result_as_dict_is_complete_and_json_safe() -> None:
    result = DialogResult(
        status=DialogStatus.APPROVAL_REQUIRED,
        conversation_id="conv_1",
        error="needs approval",
        error_type="ApprovalRequired",
    )
    payload = result.as_dict()

    assert payload["status"] == "approval_required"
    assert payload["ok"] is False
    assert payload["conversation_id"] == "conv_1"
    assert payload["pending"] is None
    assert payload["persist_error"] is None


def test_tool_call_record_as_dict() -> None:
    record = ToolCallRecord(
        tool="add_numbers", status="success", sequence=1, execution_id="exe_1", reason="ok"
    )
    assert record.as_dict() == {
        "tool": "add_numbers",
        "status": "success",
        "sequence": 1,
        "execution_id": "exe_1",
        "reason": "ok",
        "error": None,
    }


def test_every_status_has_a_string_value() -> None:
    assert {status.value for status in DialogStatus} >= {
        "success",
        "approval_required",
        "tool_limit_reached",
        "invalid_tool_call",
        "ai_error",
        "memory_error",
        "tool_error",
        "empty_response",
        "pending_expired",
        "disabled",
    }
