"""Tool-call protocol between the model and the orchestrator (Phase 5).

Two channels exist, and both are explicit:

1. **Structured** (preferred, always enabled): the tool call arrives in
   `AIResponse.metadata["tool_call"]`. Nothing is parsed out of prose. The
   Ollama and OpenAI-compatible providers fill this key from their native
   `tool_calls` response field (`robanayt_ai.ai.tool_calls` normalises it and drops
   everything except `name` and `arguments`); any future provider adapter can
   fill the same key.

2. **Text** (opt-in, disabled by default): a fenced ```` ```json ```` block
   holding exactly one object with one key, `tool_call`. A block that is not
   valid JSON is reported as an error instead of being executed.

No attempt is made to guess an intent from ordinary prose: if the model just
writes "I could add 2 and 2", that is an answer, not a call. Whatever channel
the request comes from, it is validated by `robanayt_ai.tools.bridge`, so a model
cannot smuggle a permission decision into the payload, and it is executed only
by the shared `ToolExecutor` after the policy check.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from robanayt_ai.ai.models import AIResponse, AITool
from robanayt_ai.ai.tool_calls import TOOL_CALL_KEY
from robanayt_ai.tools.bridge import tool_descriptions, tool_call_from_ai
from robanayt_ai.tools.errors import ToolValidationError
from robanayt_ai.tools.registry import ToolRegistry

_JSON_FENCE = re.compile(r"```json\s*(.+?)```", re.DOTALL)

# Tool argument kinds -> JSON Schema types understood by both providers.
_JSON_SCHEMA_TYPES = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
}

SOURCE_METADATA = "metadata"
SOURCE_TEXT = "text"


@dataclass(frozen=True)
class ModelToolCall:
    """A validated tool request proposed by a model."""

    name: str
    arguments: dict[str, Any]
    source: str


def tool_declarations(registry: ToolRegistry) -> tuple[AITool, ...]:
    """Registered tools as provider-neutral schemas for an AI request.

    The AI layer only learns *that* a tool exists and what it accepts; the
    permission level never becomes part of the schema, and declaring a tool
    never grants the right to call it.
    """
    declarations: list[AITool] = []
    for item in tool_descriptions(registry):
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        properties: dict[str, Any] = {}
        required: list[str] = []
        for argument in item.get("arguments") or ():
            argument_name = argument.get("name")
            if not isinstance(argument_name, str) or not argument_name:
                continue
            properties[argument_name] = _parameter_schema(argument)
            if argument.get("required"):
                required.append(argument_name)
        declarations.append(
            AITool(
                name=name,
                description=str(item.get("description") or ""),
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )
        )
    return tuple(declarations)


def _parameter_schema(argument: Mapping[str, Any]) -> dict[str, Any]:
    """One argument as a JSON-Schema fragment."""
    kind = str(argument.get("kind") or "string")
    schema: dict[str, Any] = {"type": _JSON_SCHEMA_TYPES.get(kind, "string")}
    description = argument.get("description")
    if isinstance(description, str) and description.strip():
        schema["description"] = description.strip()
    choices = argument.get("choices")
    if choices:
        schema["enum"] = list(choices)
    for source, target in (("minimum", "minimum"), ("maximum", "maximum")):
        value = argument.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            schema[target] = value
    max_length = argument.get("max_length")
    if isinstance(max_length, int) and not isinstance(max_length, bool) and max_length > 0:
        schema["maxLength"] = max_length
    return schema


def tool_call_from_response(
    response: AIResponse,
    *,
    allow_text: bool = False,
) -> tuple[ModelToolCall | None, str | None]:
    """Return `(call, error)`.

    - `call` is None when the model produced a normal answer.
    - `error` is set only when an explicit tool-call block was malformed; the
      caller must then stop the turn instead of executing anything.
    """
    metadata = getattr(response, "metadata", None) or {}
    if isinstance(metadata, Mapping) and TOOL_CALL_KEY in metadata:
        try:
            request = tool_call_from_ai(metadata[TOOL_CALL_KEY], caller="model")
        except ToolValidationError as exc:
            return None, f"malformed structured tool call: {exc}"
        return (
            ModelToolCall(name=request.tool, arguments=dict(request.arguments), source=SOURCE_METADATA),
            None,
        )

    if not allow_text:
        return None, None
    return _from_text(getattr(response, "content", "") or "")


def _from_text(content: str) -> tuple[ModelToolCall | None, str | None]:
    for match in _JSON_FENCE.finditer(content):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"malformed tool call block: {exc.msg}"
        if isinstance(payload, Mapping) and TOOL_CALL_KEY in payload:
            try:
                request = tool_call_from_ai(payload[TOOL_CALL_KEY], caller="model")
            except ToolValidationError as exc:
                return None, f"malformed tool call: {exc}"
            return (
                ModelToolCall(
                    name=request.tool, arguments=dict(request.arguments), source=SOURCE_TEXT
                ),
                None,
            )
    return None, None


def text_protocol_example(tool: str = "add_numbers") -> str:
    """The exact text format, for prompts and documentation."""
    return (
        "```json\n"
        + json.dumps(
            {TOOL_CALL_KEY: {"name": tool, "arguments": {"a": 1, "b": 2}}},
            ensure_ascii=False,
            indent=2,
        )
        + "\n```"
    )
