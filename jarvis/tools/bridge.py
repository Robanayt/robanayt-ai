"""Bridge between the AI layer and the tool system (Phase 4).

The AI layer may *describe* tools and may *propose* a call. It may never
grant permission: a tool call arriving from a model is treated as untrusted
data, validated like any other input, and then handed to the single executor
which applies the policy again.

This module performs no I/O and imports no provider SDK.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jarvis.tools.errors import ToolValidationError
from jarvis.tools.models import ToolRequest
from jarvis.tools.registry import ToolRegistry

# A model must not be able to smuggle a permission decision through a payload.
_FORBIDDEN_KEYS = frozenset(
    {
        "approved",
        "approve",
        "approval",
        "confirmed",
        "confirm",
        "permission",
        "force",
        "bypass",
        "elevated",
        "sudo",
        "trusted",
    }
)

_ALLOWED_KEYS = frozenset({"name", "arguments", "caller"})


def tool_descriptions(registry: ToolRegistry) -> tuple[dict[str, Any], ...]:
    """Provider-neutral tool descriptions for an AI request."""
    return tuple(registry.describe())


def tool_call_from_ai(payload: Mapping[str, Any], *, caller: str = "ai") -> ToolRequest:
    """Validate a tool call proposed by a model.

    Returns a `ToolRequest`; raises `ToolValidationError` for anything
    unexpected. The returned request carries no permission - the executor
    decides that later.
    """
    if not isinstance(payload, Mapping):
        raise ToolValidationError("tool call must be a mapping")
    forbidden = sorted(set(payload) & _FORBIDDEN_KEYS)
    if forbidden:
        raise ToolValidationError(
            f"a model cannot grant permission: forbidden field(s): {', '.join(forbidden)}"
        )
    unknown = sorted(set(payload) - _ALLOWED_KEYS)
    if unknown:
        raise ToolValidationError(
            f"unsupported tool call field(s): {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(_ALLOWED_KEYS))}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolValidationError("tool call requires a non-empty 'name'")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("tool call 'arguments' must be a mapping")
    return ToolRequest(
        tool=name.strip(),
        arguments=dict(arguments),
        caller=str(payload.get("caller") or caller),
    )
