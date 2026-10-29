"""JARVIS_AI dialog orchestration (Phase 5).

A single, bounded dialogue cycle that connects the intelligence layer
(Phase 2), memory (Phase 3) and the tool system (Phase 4):

    message -> conversation -> limited context -> AI -> (validated tool call,
    executed only with permission) -> follow-up -> final answer -> persist

This layer owns no database, no provider client and no tool registry: it
reuses `AIService`, `MemoryService` and `ToolSystem`. It never starts a
network request by itself, never approves a tool call on its own and never
loops without a hard limit (`max_tool_calls_per_message`, default 3).
"""

from jarvis.orchestration.errors import (
    OrchestrationDisabledError,
    OrchestrationError,
    PendingApprovalError,
    ToolCallProtocolError,
)
from jarvis.orchestration.models import (
    DEFAULT_MEMORY_SNIPPETS,
    MAX_MESSAGE_CHARS,
    MAX_PENDING_TURNS,
    MAX_TOOL_RESULT_CHARS,
    DialogResult,
    DialogStatus,
    PendingApproval,
    ToolCallRecord,
)
from jarvis.orchestration.protocol import (
    TOOL_CALL_KEY,
    ModelToolCall,
    tool_call_from_response,
)
from jarvis.orchestration.prompts import build_system_prompt, build_tool_guidance
from jarvis.orchestration.service import DialogService
from jarvis.orchestration.turn import TurnCancellation, TurnObserver

__all__ = [
    "DEFAULT_MEMORY_SNIPPETS",
    "MAX_MESSAGE_CHARS",
    "MAX_PENDING_TURNS",
    "MAX_TOOL_RESULT_CHARS",
    "TOOL_CALL_KEY",
    "DialogResult",
    "DialogService",
    "DialogStatus",
    "ModelToolCall",
    "OrchestrationDisabledError",
    "OrchestrationError",
    "PendingApproval",
    "PendingApprovalError",
    "ToolCallProtocolError",
    "ToolCallRecord",
    "TurnCancellation",
    "TurnObserver",
    "build_system_prompt",
    "build_tool_guidance",
    "tool_call_from_response",
]
