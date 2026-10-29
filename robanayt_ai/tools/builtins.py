"""Safe demonstration tools (Phase 4).

These three tools exist to prove the whole loop works end to end
(registration -> validation -> permission -> execution -> result -> audit)
without any external side effect:

- `current_time`: reads the clock through the injected `now()`;
- `add_numbers`: pure arithmetic with validated finite numbers;
- `list_tools`: reads the registry - it cannot change it.

Nothing here touches the filesystem, the network, the clipboard, running
processes or system settings. There is deliberately no shell, no `eval`, no
file deletion and no browser or OS automation in this phase.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from robanayt_ai.tools.models import (
    ArgumentSpec,
    PermissionLevel,
    ToolContext,
    ToolHandler,
    ToolSpec,
)
from robanayt_ai.tools.policy import PermissionPolicy
from robanayt_ai.tools.registry import ToolRegistry

BUILTIN_NAMES = ("add_numbers", "current_time", "echo_text", "list_tools")
SAFE_BUILTIN_NAMES = ("add_numbers", "current_time", "list_tools")


def current_time_spec() -> ToolSpec:
    return ToolSpec(
        name="current_time",
        description="Return the current UTC time (ISO-8601 and epoch seconds).",
        version="1.0.0",
        arguments=(),
        permission=PermissionLevel.SAFE,
        side_effects=False,
    )


def current_time_handler(arguments: Any, context: ToolContext) -> dict[str, Any]:
    epoch = float(context.now())
    moment = _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc)
    return {
        "iso": moment.isoformat().replace("+00:00", "Z"),
        "epoch": epoch,
        "timezone": "UTC",
    }


def add_numbers_spec() -> ToolSpec:
    return ToolSpec(
        name="add_numbers",
        description="Add two finite numbers and return the sum.",
        version="1.0.0",
        arguments=(
            ArgumentSpec(name="a", kind="number", description="first addend"),
            ArgumentSpec(name="b", kind="number", description="second addend"),
        ),
        permission=PermissionLevel.SAFE,
        side_effects=False,
    )


def add_numbers_handler(arguments: Any, context: ToolContext) -> dict[str, Any]:
    a = float(arguments["a"])
    b = float(arguments["b"])
    return {"sum": a + b, "expression": f"{a:g} + {b:g}"}


def echo_text_spec() -> ToolSpec:
    return ToolSpec(
        name="echo_text",
        description=(
            "Return the given text unchanged. No external side effect, but the "
            "policy still requires an explicit approval bound to the exact text."
        ),
        version="1.0.0",
        arguments=(
            ArgumentSpec(
                name="text",
                kind="string",
                description="text to echo back",
                max_length=200,
            ),
        ),
        # No side effects, yet deliberately not SAFE: it demonstrates the
        # approval flow end to end.
        permission=PermissionLevel.APPROVAL,
        side_effects=False,
    )


def echo_text_handler(arguments: Any, context: ToolContext) -> dict[str, Any]:
    return {"text": str(arguments["text"]), "length": len(str(arguments["text"]))}


def list_tools_spec() -> ToolSpec:
    return ToolSpec(
        name="list_tools",
        description="List the registered tools with version and permission level.",
        version="1.0.0",
        arguments=(),
        permission=PermissionLevel.SAFE,
        side_effects=False,
    )


def build_list_tools_handler(registry: ToolRegistry) -> ToolHandler:
    """Return a handler bound to a registry (still read-only)."""

    def handler(arguments: Any, context: ToolContext) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "version": spec.version,
                "description": spec.description,
                "permission": spec.permission.value,
                "arguments": len(spec.arguments),
            }
            for spec in registry.specs()
        ]

    return handler


def register_builtins(registry: ToolRegistry) -> tuple[str, ...]:
    """Register the demo tools. Registration never executes them."""
    registry.register(current_time_spec(), current_time_handler)
    registry.register(add_numbers_spec(), add_numbers_handler)
    registry.register(echo_text_spec(), echo_text_handler)
    registry.register(list_tools_spec(), build_list_tools_handler(registry))
    return BUILTIN_NAMES


def builtin_policy(**kwargs: Any) -> PermissionPolicy:
    """Default policy: read-only demo tools are safe, `echo_text` needs an
    approval, everything else is unknown and therefore denied."""
    known: dict[str, PermissionLevel] = {
        name: PermissionLevel.SAFE for name in SAFE_BUILTIN_NAMES
    }
    known["echo_text"] = PermissionLevel.APPROVAL
    known.update(kwargs.pop("known", {}))
    return PermissionPolicy(known, **kwargs)
