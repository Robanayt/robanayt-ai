"""Dialog orchestration errors (Phase 5).

The base class is `OrchestrationError`, so callers can catch the whole layer.
Messages are built from names, statuses and counts: message text, tool
arguments and provider payloads never appear in an exception message.
"""

from __future__ import annotations


class OrchestrationError(Exception):
    """Base class for every dialog-orchestration failure."""


class OrchestrationDisabledError(OrchestrationError):
    """The orchestration layer is disabled by configuration."""


class ToolCallProtocolError(OrchestrationError):
    """A tool call from the model was present but malformed."""


class PendingApprovalError(OrchestrationError):
    """An approval was requested for a pending call that does not exist."""
