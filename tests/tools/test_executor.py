"""Executor pipeline: lookup -> validation -> permission -> run -> audit."""

from __future__ import annotations

from typing import Any

import pytest

from tests.tools.helpers import (
    FakeClock,
    PermissionLevel,
    approval_spec,
    counting_handler,
    denied_spec,
    raising_handler,
    safe_spec,
    sleeping_handler,
)
from jarvis.tools.audit import AuditLog
from jarvis.tools.errors import ToolValidationError
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.models import (
    ArgumentSpec,
    PermissionDecision,
    PermissionLevel as Level,
    ToolContext,
    ToolRequest,
    ToolStatus,
    digest_of,
)
from jarvis.tools.policy import PermissionPolicy
from jarvis.tools.registry import ToolRegistry


def build(
    clock: FakeClock,
    entries: list[tuple[Any, Any]],
    known: dict[str, Level] | None = None,
    *,
    default_timeout: float = 5.0,
    enabled: bool = True,
):
    registry = ToolRegistry()
    for spec, handler in entries:
        registry.register(spec, handler)
    policy = PermissionPolicy(known or {}, clock=clock)
    audit = AuditLog(clock=clock)
    executor = ToolExecutor(
        registry,
        policy,
        audit=audit,
        default_timeout=default_timeout,
        clock=clock,
        enabled=enabled,
    )
    return registry, policy, audit, executor


def test_safe_tool_runs_and_returns_structured_result() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    *_, executor = build(clock, [(safe_spec("ping"), counting_handler(calls, "pong"))])

    request = ToolRequest(tool="ping", arguments={})
    result = executor.execute(request)

    assert result.status is ToolStatus.SUCCESS
    assert result.ok is True
    assert result.output == "pong"
    assert calls == [{}]
    assert result.request_id == request.request_id


def test_unknown_tool_is_not_executed() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    *_, audit, executor = build(clock, [(safe_spec("ping"), counting_handler(calls))])

    result = executor.execute(ToolRequest(tool="nope", arguments={}))

    assert result.status is ToolStatus.NOT_FOUND
    assert calls == []
    assert len(audit) == 1
    assert audit.last().status == "not_found"


def test_invalid_arguments_are_rejected_before_execution() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    spec = safe_spec("ping", arguments=(ArgumentSpec(name="a", kind="integer"),))
    *_, executor = build(clock, [(spec, counting_handler(calls))])

    result = executor.execute(ToolRequest(tool="ping", arguments={"a": "not a number"}))

    assert result.status is ToolStatus.INVALID
    assert result.error_type == "ToolValidationError"
    assert calls == []


def test_denied_tool_never_runs() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    *_, executor = build(clock, [(denied_spec(), counting_handler(calls))])

    result = executor.execute(ToolRequest(tool="forbidden", arguments={}))

    assert result.status is ToolStatus.DENIED
    assert result.permission == "denied"
    assert calls == []


def test_approval_required_tool_waits_for_approval() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    spec = approval_spec(
        "touch", arguments=(ArgumentSpec(name="path", kind="string"),)
    )
    _, policy, audit, executor = build(
        clock, [(spec, counting_handler(calls))], {"touch": Level.APPROVAL}
    )

    request = ToolRequest(tool="touch", arguments={"path": "a.txt"})
    result = executor.execute(request)

    assert result.status is ToolStatus.APPROVAL_REQUIRED
    assert calls == []
    assert audit.last().status == "approval_required"

    approval = policy.grant(spec, request.arguments_digest())
    approved = executor.execute(request, approval=approval)
    assert approved.status is ToolStatus.SUCCESS
    assert calls == [{"path": "a.txt"}]


def test_approval_is_bound_to_the_exact_arguments() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    spec = approval_spec(
        "touch", arguments=(ArgumentSpec(name="path", kind="string"),)
    )
    _, policy, _, executor = build(
        clock, [(spec, counting_handler(calls))], {"touch": Level.APPROVAL}
    )

    approval = policy.grant(spec, digest_of({"path": "a.txt"}))
    changed = ToolRequest(tool="touch", arguments={"path": "b.txt"})
    result = executor.execute(changed, approval=approval)

    assert result.status is ToolStatus.APPROVAL_REQUIRED
    assert "arguments changed" in result.reason
    assert calls == []


def test_approval_cannot_be_reused() -> None:
    clock = FakeClock()
    spec = approval_spec("touch")
    _, policy, _, executor = build(
        clock, [(spec, counting_handler([]))], {"touch": Level.APPROVAL}
    )

    request = ToolRequest(tool="touch", arguments={})
    approval = policy.grant(spec, request.arguments_digest())

    assert executor.execute(request, approval=approval).status is ToolStatus.SUCCESS
    second = executor.execute(request, approval=approval)
    assert second.status is ToolStatus.APPROVAL_REQUIRED
    assert "already been used" in second.reason


def test_handler_exception_becomes_a_failed_result() -> None:
    clock = FakeClock()
    calls: list[int] = []
    *_, audit, executor = build(
        clock, [(safe_spec("boom"), raising_handler("kaboom", calls))]
    )

    result = executor.execute(ToolRequest(tool="boom", arguments={}))

    assert result.status is ToolStatus.FAILED
    assert result.error == "kaboom"
    assert result.error_type == "RuntimeError"
    assert result.ok is False
    assert calls == [1]  # never retried
    assert len(audit) == 1


def test_slow_tool_times_out() -> None:
    clock = FakeClock()
    *_, audit, executor = build(
        clock, [(safe_spec("slow"), sleeping_handler(0.6))], default_timeout=0.05
    )

    result = executor.execute(ToolRequest(tool="slow", arguments={}))

    assert result.status is ToolStatus.TIMEOUT
    assert result.output is None
    assert "exceeded" in (result.error or "")
    assert audit.last().status == "timeout"


def test_cooperative_handler_can_stop_before_the_deadline() -> None:
    clock = FakeClock()

    def handler(arguments: Any, context: ToolContext) -> str:
        return "stopped early" if context.expired() else "ran"

    *_, executor = build(clock, [(safe_spec("fast"), handler)], default_timeout=0.05)
    result = executor.execute(ToolRequest(tool="fast", arguments={}))

    assert result.status is ToolStatus.SUCCESS


def test_disabled_executor_denies_everything() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    *_, executor = build(
        clock, [(safe_spec("ping"), counting_handler(calls))], enabled=False
    )

    result = executor.execute(ToolRequest(tool="ping", arguments={}))

    assert result.status is ToolStatus.DENIED
    assert "disabled" in result.reason
    assert calls == []


def test_policy_is_checked_on_every_execution() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    _, policy, _, executor = build(
        clock, [(safe_spec("ping"), counting_handler(calls))], {"ping": Level.SAFE}
    )
    original = policy.evaluate
    seen: list[str] = []

    def evaluate(spec: Any) -> PermissionDecision:
        seen.append(spec.name)
        if len(seen) > 1:
            return PermissionDecision(Level.DENIED, False, False, "revoked after review")
        return original(spec)

    policy.evaluate = evaluate  # type: ignore[method-assign]

    assert executor.execute(ToolRequest(tool="ping")).status is ToolStatus.SUCCESS
    second = executor.execute(ToolRequest(tool="ping"))
    assert second.status is ToolStatus.DENIED
    assert second.reason == "revoked after review"
    assert calls == [{}]  # only the allowed run executed


def test_every_attempt_is_audited_without_argument_values() -> None:
    clock = FakeClock()
    spec = safe_spec("echo", arguments=(ArgumentSpec(name="secret", kind="string"),))
    *_, audit, executor = build(clock, [(spec, counting_handler([], "ok"))])

    executor.execute(ToolRequest(tool="echo", arguments={"secret": "hunter2"}))
    executor.execute(ToolRequest(tool="missing", arguments={}))

    assert len(audit) == 2
    payload = audit.to_json()
    assert "hunter2" not in payload
    assert audit.records()[0].argument_keys == ("secret",)
    assert audit.records()[0].permission == "safe"


def test_audit_records_permission_reason_and_duration() -> None:
    clock = FakeClock()
    *_, audit, executor = build(clock, [(denied_spec(), counting_handler([]))])

    executor.execute(ToolRequest(tool="forbidden", arguments={}))
    record = audit.last()

    assert record.status == "denied"
    assert record.permission == "denied"
    assert record.reason
    assert record.duration_ms >= 0


def test_executor_does_not_retry_side_effects() -> None:
    clock = FakeClock()
    calls: list[int] = []
    spec = approval_spec(
        "charge", arguments=(ArgumentSpec(name="amount", kind="number"),)
    )
    _, policy, _, executor = build(
        clock, [(spec, raising_handler("gateway down", calls))], {"charge": Level.APPROVAL}
    )
    request = ToolRequest(tool="charge", arguments={"amount": 10})
    approval = policy.grant(spec, request.arguments_digest())

    result = executor.execute(request, approval=approval)

    assert result.status is ToolStatus.FAILED
    assert calls == [1]  # exactly one attempt, never repeated


def test_tool_call_from_model_still_requires_permission() -> None:
    clock = FakeClock()
    calls: list[dict[str, Any]] = []
    spec = approval_spec("touch")
    *_, executor = build(clock, [(spec, counting_handler(calls))])  # unknown -> denied

    result = executor.execute(ToolRequest(tool="touch", arguments={}))

    assert result.status is ToolStatus.DENIED
    assert calls == []


def test_validation_error_type_is_exported_for_callers() -> None:
    assert issubclass(ToolValidationError, Exception)
