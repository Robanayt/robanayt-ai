"""Provider-neutral tool-call plumbing for the AI layer.

Every provider translates its own wire format into one shared shape and
publishes it where the orchestrator already looks for it:

    AIResponse.metadata["tool_call"] == {"name": <str>, "arguments": {..}}

Rules enforced here:

- Only `name` and `arguments` survive normalisation. Provider-specific extras
  (ids, index, call type, permission-looking fields) are dropped, so a model
  can never smuggle a permission decision through a provider payload.
- At most **one** call per response is accepted. Several calls in one response
  are ambiguous for a bounded orchestrator and are rejected, not guessed.
- A malformed, unsupported or ambiguous call is *dropped*, never executed and
  never turned into a provider error: the ordinary text answer is preserved.
- This module performs no validation of tool names against a registry and no
  permission decision - that stays in `jarvis.tools.bridge` and
  `jarvis.tools.policy`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from jarvis.ai.models import AITool

logger = logging.getLogger(__name__)

# Key under which a structured tool call travels to the orchestrator.
TOOL_CALL_KEY = "tool_call"

# Only function-style calls are understood; anything else is unsupported.
_SUPPORTED_CALL_TYPES = frozenset({"function"})


def normalize_arguments(value: Any) -> dict[str, Any] | None:
    """Return arguments as a JSON object, or None when unusable.

    Accepts a mapping (used by Ollama) and a JSON object encoded as a string
    (used by OpenAI-compatible endpoints). A JSON string holding anything but
    an object is rejected.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def normalize_tool_call(payload: Any) -> dict[str, Any] | None:
    """Normalise one provider tool call, or return None to reject it.

    Both shapes are accepted: `{"name": .., "arguments": ..}` and the nested
    `{"type": "function", "function": {"name": .., "arguments": ..}}` used by
    Ollama and OpenAI-compatible endpoints.
    """
    if not isinstance(payload, Mapping):
        return None
    declared_type = payload.get("type")
    if declared_type is not None and str(declared_type).strip().lower() not in _SUPPORTED_CALL_TYPES:
        return None
    call = payload.get("function") if isinstance(payload.get("function"), Mapping) else payload
    if not isinstance(call, Mapping):
        return None
    name = call.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = normalize_arguments(call.get("arguments"))
    if arguments is None:
        return None
    # Only these two fields ever leave this function.
    return {"name": name.strip(), "arguments": arguments}


def tool_call_from_message(message: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Extract at most one tool call from a provider message.

    Returns `(call, rejection_reason)`. `rejection_reason` is informational:
    the caller logs it and keeps the ordinary response. Nothing is raised and
    nothing is executed.
    """
    if not isinstance(message, Mapping):
        return None, None
    raw = message.get("tool_calls")
    if raw is None:
        return None, None
    if not isinstance(raw, (list, tuple)):
        return None, "provider returned a malformed 'tool_calls' field"
    if not raw:
        return None, None
    if len(raw) > 1:
        return (
            None,
            f"provider returned {len(raw)} tool calls; one call per response is supported",
        )
    call = normalize_tool_call(raw[0])
    if call is None:
        return None, "provider returned an unusable tool call"
    return call, None


def tool_call_metadata(
    call: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Response metadata carrying the structured call (or nothing)."""
    result: dict[str, Any] = dict(metadata or {})
    if call is not None:
        result[TOOL_CALL_KEY] = {
            "name": str(call.get("name") or ""),
            "arguments": dict(call.get("arguments") or {}),
        }
    return result


def function_schema(tool: AITool) -> dict[str, Any]:
    """One tool in the `function` wire format shared by both providers."""
    return {"type": "function", "function": tool.as_dict()}


def tool_payload(tools: Iterable[AITool]) -> list[dict[str, Any]]:
    """Request payload fragment for declared tools (empty when none)."""
    return [function_schema(tool) for tool in tools]
