"""Tool system errors (Phase 4).

The base class is `ToolError`, so callers can catch the whole subsystem
without guessing. Nothing here ever carries an argument value: messages are
built from names, types and counts only, because arguments may contain
private user data.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for every tool-system failure."""


class ToolDisabledError(ToolError):
    """The tool subsystem is disabled by configuration."""


class ToolSpecError(ToolError):
    """A tool or argument specification is malformed."""


class DuplicateToolError(ToolError):
    """A tool with the same name is already registered."""


class ToolNotFoundError(ToolError):
    """No tool is registered under the requested name."""


class ToolValidationError(ToolError):
    """Arguments failed validation; the tool was not executed."""


class ToolPermissionError(ToolError):
    """The policy forbids this tool for this caller."""


class ApprovalRequiredError(ToolError):
    """The tool may run, but only with a valid approval token."""


class ToolExecutionError(ToolError):
    """The tool raised during execution."""


class ToolTimeoutError(ToolError):
    """The tool did not finish within the allowed time."""
