"""JARVIS_AI tool system (Phase 4).

Provider-independent tool layer: a registry of validated tool contracts, a
central permission policy with argument-bound approvals, a single executor
and an audit log that never stores argument values or secrets.

The AI layer may describe tools and propose calls (`jarvis.tools.bridge`),
but execution always goes through `ToolExecutor`, which re-checks the policy
on every call. Nothing here starts a network request, opens a database or
runs a tool as a side effect of import or startup.
"""

from jarvis.tools.audit import TOOL_BLOCKED, TOOL_EXECUTED, AuditLog, scrub
from jarvis.tools.errors import (
    ApprovalRequiredError,
    DuplicateToolError,
    ToolDisabledError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolSpecError,
    ToolTimeoutError,
    ToolValidationError,
)
from jarvis.tools.executor import DEFAULT_TIMEOUT_SECONDS, ToolExecutor
from jarvis.tools.models import (
    ApprovalToken,
    ArgumentSpec,
    AuditRecord,
    PermissionDecision,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolHandler,
    ToolRequest,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from jarvis.tools.policy import DEFAULT_APPROVAL_TTL, PermissionPolicy
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.system import ToolSystem, build_tool_system

__all__ = [
    "DEFAULT_APPROVAL_TTL",
    "DEFAULT_TIMEOUT_SECONDS",
    "TOOL_BLOCKED",
    "TOOL_EXECUTED",
    "ApprovalRequiredError",
    "ApprovalToken",
    "ArgumentSpec",
    "AuditLog",
    "AuditRecord",
    "DuplicateToolError",
    "PermissionDecision",
    "PermissionLevel",
    "PermissionPolicy",
    "Tool",
    "ToolContext",
    "ToolDisabledError",
    "ToolError",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolHandler",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolRegistry",
    "ToolRequest",
    "ToolResult",
    "ToolSpec",
    "ToolSpecError",
    "ToolStatus",
    "ToolSystem",
    "ToolTimeoutError",
    "ToolValidationError",
    "build_tool_system",
    "scrub",
]
