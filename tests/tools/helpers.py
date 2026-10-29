"""Shared helpers for tool-system tests: fake clocks, fake tools, ready systems.

Imported by package path (`from tests.tools.helpers import ...`), so no test
directory has to patch `sys.path`. That patching is what made a bare
`import conftest` ambiguous between `tests/ai/conftest.py` and
`tests/tools/conftest.py` when both suites were collected in one session.
"""

from __future__ import annotations

from typing import Any

from jarvis.core.config import ToolsConfig
from jarvis.tools.models import (
    ArgumentSpec,
    PermissionLevel,
    ToolContext,
    ToolSpec,
)
from jarvis.tools.policy import PermissionPolicy
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.system import ToolSystem, build_tool_system


class FakeClock:
    """Deterministic clock: only advances when a test says so."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def safe_spec(name: str = "ping", **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"safe test tool {name}",
        permission=PermissionLevel.SAFE,
        side_effects=False,
        **kwargs,
    )


def approval_spec(name: str = "touch", **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"approval-required test tool {name}",
        permission=PermissionLevel.APPROVAL,
        side_effects=True,
        **kwargs,
    )


def denied_spec(name: str = "forbidden") -> ToolSpec:
    return ToolSpec(
        name="forbidden",
        description="forbidden test tool",
        permission=PermissionLevel.DENIED,
        side_effects=True,
    )


def counting_handler(calls: list[dict[str, Any]], result: Any = "done"):
    """Handler that records every invocation - never called implicitly."""

    def handler(arguments: Any, context: ToolContext) -> Any:
        calls.append(dict(arguments))
        return result

    return handler


def raising_handler(message: str = "boom", calls: list[int] | None = None):
    def handler(arguments: Any, context: ToolContext) -> Any:
        if calls is not None:
            calls.append(1)
        raise RuntimeError(message)

    return handler


def sleeping_handler(seconds: float = 0.5):
    import time as _time

    def handler(arguments: Any, context: ToolContext) -> Any:
        _time.sleep(seconds)
        return "late"

    return handler


def build_system(
    *,
    clock: FakeClock | None = None,
    tools_config: ToolsConfig | None = None,
    with_builtins: bool = False,
) -> ToolSystem:
    """Build a tool system with an injected clock (no I/O, no network)."""
    return build_tool_system(
        tools_config or ToolsConfig(),
        clock=clock or FakeClock(),
        with_builtins=with_builtins,
    )


def empty_registry_policy(
    clock: FakeClock | None = None,
) -> tuple[ToolRegistry, PermissionPolicy]:
    return ToolRegistry(), PermissionPolicy(clock=clock or FakeClock())


__all__ = [
    "ArgumentSpec",
    "FakeClock",
    "PermissionLevel",
    "ToolSpec",
    "approval_spec",
    "build_system",
    "counting_handler",
    "denied_spec",
    "empty_registry_policy",
    "raising_handler",
    "safe_spec",
    "sleeping_handler",
]
