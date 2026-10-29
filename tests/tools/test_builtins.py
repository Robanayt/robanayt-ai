"""Safe demo tools: registration never runs them, results are deterministic."""

from __future__ import annotations

import datetime as dt

import pytest

from tests.tools.helpers import FakeClock
from robanayt_ai.core.config import ToolsConfig
from robanayt_ai.tools import build_tool_system
from robanayt_ai.tools.builtins import (
    BUILTIN_NAMES,
    SAFE_BUILTIN_NAMES,
    builtin_policy,
    register_builtins,
)
from robanayt_ai.tools.models import PermissionLevel, ToolRequest, ToolStatus
from robanayt_ai.tools.registry import ToolRegistry


def test_registration_adds_all_builtin_tools() -> None:
    registry = ToolRegistry()
    register_builtins(registry)

    assert registry.names() == BUILTIN_NAMES


def test_builtin_policy_marks_safe_tools_and_the_approval_demo() -> None:
    policy = builtin_policy(clock=FakeClock())

    assert policy.known()["current_time"] == "safe"
    assert policy.known()["add_numbers"] == "safe"
    assert policy.known()["list_tools"] == "safe"
    assert policy.known()["echo_text"] == "approval_required"


def test_current_time_uses_the_injected_clock() -> None:
    clock = FakeClock(1_700_000_000.0)
    system = build_tool_system(ToolsConfig(), clock=clock)

    result = system.executor.execute(ToolRequest(tool="current_time"))

    assert result.status is ToolStatus.SUCCESS
    expected = dt.datetime.fromtimestamp(1_700_000_000.0, tz=dt.timezone.utc)
    assert result.output["iso"] == expected.isoformat().replace("+00:00", "Z")
    assert result.output["epoch"] == 1_700_000_000.0


def test_add_numbers_validates_and_sums() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())

    result = system.executor.execute(
        ToolRequest(tool="add_numbers", arguments={"a": 2, "b": 3.5})
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.output["sum"] == 5.5


def test_add_numbers_rejects_bad_input() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())

    result = system.executor.execute(
        ToolRequest(tool="add_numbers", arguments={"a": "two", "b": 1})
    )

    assert result.status is ToolStatus.INVALID


def test_add_numbers_rejects_booleans() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())

    result = system.executor.execute(
        ToolRequest(tool="add_numbers", arguments={"a": True, "b": 1})
    )

    assert result.status is ToolStatus.INVALID


def test_list_tools_reads_the_registry() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())

    result = system.executor.execute(ToolRequest(tool="list_tools"))

    assert result.status is ToolStatus.SUCCESS
    assert {item["name"] for item in result.output} == set(BUILTIN_NAMES)


def test_echo_text_requires_approval_before_running() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())
    request = ToolRequest(tool="echo_text", arguments={"text": "hello"})

    blocked = system.executor.execute(request)
    assert blocked.status is ToolStatus.APPROVAL_REQUIRED

    approval = system.policy.grant(
        system.registry.require("echo_text").spec, request.arguments_digest()
    )
    allowed = system.executor.execute(request, approval=approval)
    assert allowed.status is ToolStatus.SUCCESS
    assert allowed.output["text"] == "hello"


def test_echo_text_audit_never_contains_the_text() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())
    request = ToolRequest(tool="echo_text", arguments={"text": "private note"})
    approval = system.policy.grant(
        system.registry.require("echo_text").spec, request.arguments_digest()
    )

    system.executor.execute(request, approval=approval)

    assert "private note" not in system.audit.to_json()


def test_safe_builtins_are_declared_without_side_effects() -> None:
    registry = ToolRegistry()
    register_builtins(registry)

    for name in SAFE_BUILTIN_NAMES:
        spec = registry.require(name).spec
        assert spec.permission is PermissionLevel.SAFE
        assert spec.side_effects is False


def test_echo_text_rejects_oversized_text() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())

    result = system.executor.execute(
        ToolRequest(tool="echo_text", arguments={"text": "x" * 500})
    )

    assert result.status is ToolStatus.INVALID


def test_building_the_system_executes_nothing() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())

    assert system.audit.records() == ()
    assert len(system.registry) == len(BUILTIN_NAMES)
