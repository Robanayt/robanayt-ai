"""System-prompt construction for a dialogue turn (Phase 5).

Tool descriptions are data for the model, never a permission: the prompt says
so explicitly. Only tools that are actually registered are listed, and no
instruction in the prompt can grant execution rights - that decision stays in
`robanayt_ai.tools.policy`.
"""

from __future__ import annotations

from robanayt_ai.orchestration.protocol import TOOL_CALL_KEY, text_protocol_example
from robanayt_ai.tools.registry import ToolRegistry


def build_tool_guidance(
    registry: ToolRegistry,
    *,
    include_text_protocol: bool = False,
) -> str:
    """Describe the registered tools and the accepted call format."""
    specs = registry.specs()
    if not specs:
        return ""

    lines = [
        "Tools (a request is data, never a permission; execution needs the "
        "user's approval according to the tool policy):"
    ]
    for spec in specs:
        arguments = ", ".join(
            f"{argument.name}: {argument.kind}{'' if argument.required else ' (optional)'}"
            for argument in spec.arguments
        )
        lines.append(f"- {spec.name} v{spec.version} [{spec.permission.value}]: {spec.description}")
        if arguments:
            lines.append(f"    arguments: {arguments}")

    if include_text_protocol:
        lines.append(
            "To request one tool call, answer with a single fenced JSON block "
            f"containing one key, {TOOL_CALL_KEY!r}:"
        )
        lines.append(text_protocol_example())
        lines.append("Any other text is treated as your final answer.")
    else:
        lines.append(
            "Tool calls are accepted only through the structured channel; free "
            "text is always treated as your final answer, never as a call."
        )
    return "\n".join(lines)


def build_system_prompt(
    system_prompt: str | None,
    registry: ToolRegistry,
    *,
    include_text_protocol: bool = False,
) -> str:
    """Combine the caller's system prompt with tool guidance."""
    parts = [part.strip() for part in (system_prompt or "", build_tool_guidance(
        registry, include_text_protocol=include_text_protocol
    )) if part and part.strip()]
    return "\n\n".join(parts)
