"""Validation, digests, results and approval binding."""

from __future__ import annotations

import math

import pytest

from tests.tools.helpers import FakeClock, PermissionLevel, approval_spec, safe_spec
from jarvis.tools.models import (
    ApprovalToken,
    ArgumentSpec,
    AuditRecord,
    PermissionDecision,
    ToolRequest,
    ToolResult,
    ToolSpec,
    ToolStatus,
    digest_of,
)


def test_spec_rejects_bad_name() -> None:
    with pytest.raises(ValueError, match="tool name"):
        ToolSpec(name="Bad Name", description="x", side_effects=False)


def test_spec_requires_description() -> None:
    with pytest.raises(ValueError, match="description"):
        ToolSpec(name="ping", description="   ", side_effects=False)


def test_spec_rejects_bad_version() -> None:
    with pytest.raises(ValueError, match="version"):
        ToolSpec(name="ping", description="x", version="v1", side_effects=False)


def test_safe_spec_must_declare_no_side_effects() -> None:
    with pytest.raises(ValueError, match="side_effects"):
        ToolSpec(
            name="ping",
            description="x",
            permission=PermissionLevel.SAFE,
            side_effects=True,
        )


def test_spec_rejects_duplicate_arguments() -> None:
    with pytest.raises(ValueError, match="duplicate argument"):
        ToolSpec(
            name="ping",
            description="x",
            arguments=(
                ArgumentSpec(name="a", kind="string"),
                ArgumentSpec(name="a", kind="integer"),
            ),
            side_effects=False,
        )


def test_spec_rejects_required_after_optional() -> None:
    with pytest.raises(ValueError, match="follows an optional one"):
        ToolSpec(
            name="ping",
            description="x",
            arguments=(
                ArgumentSpec(name="a", kind="string", required=False, default=""),
                ArgumentSpec(name="b", kind="string"),
            ),
            side_effects=False,
        )


def test_argument_kind_must_be_known() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        ArgumentSpec(name="a", kind="blob")


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("integer", True),
        ("number", "1"),
        ("string", 5),
        ("boolean", "yes"),
        ("object", []),
        ("array", {}),
    ],
)
def test_argument_type_is_enforced(kind: str, value: object) -> None:
    spec = ArgumentSpec(name="a", kind=kind)
    with pytest.raises(ValueError, match="must be"):
        spec.validate_value(value)


def test_number_arguments_reject_non_finite_values() -> None:
    spec = ArgumentSpec(name="a", kind="number")
    with pytest.raises(ValueError, match="finite"):
        spec.validate_value(math.nan)
    with pytest.raises(ValueError, match="finite"):
        spec.validate_value(math.inf)


def test_argument_bounds_and_choices() -> None:
    spec = ArgumentSpec(name="a", kind="integer", minimum=1, maximum=5, choices=(1, 2, 3))
    assert spec.validate_value(2) == 2
    with pytest.raises(ValueError, match="one of"):
        spec.validate_value(4)
    bounded = ArgumentSpec(name="b", kind="number", minimum=0.0, maximum=1.0)
    with pytest.raises(ValueError, match=">="):
        bounded.validate_value(-1.0)


def test_string_max_length() -> None:
    spec = ArgumentSpec(name="a", kind="string", max_length=3)
    assert spec.validate_value("abc") == "abc"
    with pytest.raises(ValueError, match="at most 3"):
        spec.validate_value("abcd")


def test_unknown_arguments_are_rejected() -> None:
    spec = safe_spec(arguments=(ArgumentSpec(name="a", kind="string"),))
    with pytest.raises(ValueError, match="unknown argument"):
        spec.validate_arguments({"b": 1})


def test_missing_required_argument_is_rejected() -> None:
    spec = safe_spec(arguments=(ArgumentSpec(name="a", kind="string"),))
    with pytest.raises(ValueError, match="missing required argument"):
        spec.validate_arguments({})


def test_defaults_are_applied() -> None:
    spec = safe_spec(
        arguments=(
            ArgumentSpec(name="a", kind="integer", required=False, default=7),
            ArgumentSpec(name="b", kind="string", required=False, default="x"),
        )
    )
    assert spec.validate_arguments({"a": 1}) == {"a": 1, "b": "x"}


def test_non_mapping_arguments_are_rejected() -> None:
    spec = safe_spec()
    with pytest.raises(ValueError, match="mapping"):
        spec.validate_arguments(["a"])  # type: ignore[arg-type]


def test_request_digest_is_stable_and_order_independent() -> None:
    first = ToolRequest(tool="ping", arguments={"a": 1, "b": "two"})
    second = ToolRequest(tool="ping", arguments={"b": "two", "a": 1})
    assert first.arguments_digest() == second.arguments_digest()


def test_request_digest_changes_with_arguments() -> None:
    first = ToolRequest(tool="ping", arguments={"a": 1})
    second = ToolRequest(tool="ping", arguments={"a": 2})
    assert first.arguments_digest() != second.arguments_digest()


def test_request_argument_keys_are_sorted() -> None:
    request = ToolRequest(tool="ping", arguments={"b": 1, "a": 2})
    assert request.argument_keys() == ("a", "b")


def test_request_rejects_bad_tool_name() -> None:
    with pytest.raises(ValueError, match="tool name"):
        ToolRequest(tool="not valid", arguments={})


def test_result_ok_only_for_success() -> None:
    success = ToolResult(execution_id="e1", tool="ping", status=ToolStatus.SUCCESS)
    blocked = ToolResult(execution_id="e2", tool="ping", status=ToolStatus.DENIED)
    assert success.ok is True
    assert blocked.ok is False
    assert success.as_dict()["ok"] is True


def test_result_as_dict_is_json_safe() -> None:
    result = ToolResult(
        execution_id="e1",
        tool="ping",
        status=ToolStatus.SUCCESS,
        output=(1, 2),
        request_id="r1",
    )
    assert result.as_dict()["output"] == [1, 2]


def test_approval_token_is_bound_to_tool_version_and_arguments() -> None:
    clock = FakeClock()
    token = ApprovalToken(
        token_id="apr_1",
        tool="touch",
        version="1.0.0",
        arguments_digest=digest_of({"a": 1}),
        issued_at=clock(),
        expires_at=clock() + 60,
    )
    valid, reason = token.matches(
        tool="touch", version="1.0.0", digest=digest_of({"a": 1}), now=clock()
    )
    assert valid is True and reason == "valid"

    assert token.matches(tool="other", version="1.0.0", digest=digest_of({"a": 1}), now=clock())[0] is False
    assert token.matches(tool="touch", version="2.0.0", digest=digest_of({"a": 1}), now=clock())[0] is False
    assert token.matches(tool="touch", version="1.0.0", digest=digest_of({"a": 2}), now=clock())[0] is False


def test_approval_token_expires() -> None:
    clock = FakeClock()
    token = ApprovalToken(
        token_id="apr_1",
        tool="touch",
        version="1.0.0",
        arguments_digest=digest_of({}),
        issued_at=clock(),
        expires_at=clock() + 10,
    )
    clock.advance(11)
    valid, reason = token.matches(tool="touch", version="1.0.0", digest=digest_of({}), now=clock())
    assert valid is False and "expired" in reason


def test_audit_record_never_contains_argument_values() -> None:
    record = AuditRecord(
        execution_id="e1",
        tool="echo_text",
        version="1.0.0",
        timestamp=1.0,
        status="success",
        permission="safe",
        reason="ok",
        arguments_digest=digest_of({"text": "secret phrase"}),
        argument_keys=("text",),
        duration_ms=1.0,
        caller="cli",
    )
    payload = record.to_json()
    assert "secret phrase" not in payload
    assert record.as_dict()["argument_keys"] == ["text"]


def test_permission_decision_as_dict() -> None:
    decision = PermissionDecision(
        level=PermissionLevel.APPROVAL,
        allowed=False,
        requires_approval=True,
        reason="needs approval",
    )
    assert decision.as_dict() == {
        "level": "approval_required",
        "allowed": False,
        "requires_approval": True,
        "reason": "needs approval",
    }


def test_approval_spec_defaults_to_approval_level() -> None:
    assert approval_spec().permission is PermissionLevel.APPROVAL
    assert safe_spec().permission is PermissionLevel.SAFE
