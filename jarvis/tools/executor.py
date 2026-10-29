"""Single tool executor (Phase 4).

Every execution follows the same order, and every attempt produces exactly
one `ToolResult` and one audit record:

1. find the tool (unknown name -> `NOT_FOUND`, nothing runs);
2. validate arguments against the declared schema -> `INVALID`;
3. ask the policy -> `DENIED` or `APPROVAL_REQUIRED`;
4. run the handler with a deadline -> `SUCCESS`, `FAILED` or `TIMEOUT`.

There is no automatic retry: a side effect that failed must not be repeated
silently, and a slow tool is reported, not re-run.

Timeouts are enforced by a daemon thread: the executor stops waiting when the
deadline passes, but Python cannot forcibly stop a thread that is blocked in
native code. Cooperative handlers should check `ToolContext.expired()`; the
limitation is documented rather than hidden.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from jarvis.tools.audit import AuditLog, scrub
from jarvis.tools.models import (
    ApprovalToken,
    AuditRecord,
    PermissionDecision,
    PermissionLevel,
    ToolContext,
    ToolRequest,
    ToolResult,
    ToolStatus,
    new_id,
)
from jarvis.tools.policy import PermissionPolicy
from jarvis.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 5.0


class ToolExecutor:
    """The only path through which a tool may run."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        *,
        audit: AuditLog | None = None,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.time,
        enabled: bool = True,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._audit = audit
        self._default_timeout = float(default_timeout)
        self._clock = clock
        self._enabled = bool(enabled)

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def default_timeout(self) -> float:
        return self._default_timeout

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    @property
    def audit(self) -> AuditLog | None:
        return self._audit

    def execute(
        self,
        request: ToolRequest,
        *,
        approval: ApprovalToken | None = None,
        timeout: float | None = None,
    ) -> ToolResult:
        """Run one request through the full pipeline. Never raises."""
        started = self._clock()
        execution_id = new_id("exe")
        digest = request.arguments_digest()

        if not self._enabled:
            return self._finish(
                execution_id=execution_id,
                request=request,
                started=started,
                status=ToolStatus.DENIED,
                permission=PermissionLevel.DENIED,
                reason="the tool subsystem is disabled by configuration",
                digest=digest,
                version="",
            )

        tool = self._registry.get(request.tool)
        if tool is None:
            return self._finish(
                execution_id=execution_id,
                request=request,
                started=started,
                status=ToolStatus.NOT_FOUND,
                permission=PermissionLevel.DENIED,
                reason=f"unknown tool {request.tool!r}: nothing was executed",
                digest=digest,
                version="",
            )

        spec = tool.spec
        try:
            arguments = spec.validate_arguments(request.arguments)
        except ValueError as exc:
            return self._finish(
                execution_id=execution_id,
                request=request,
                started=started,
                status=ToolStatus.INVALID,
                permission=PermissionLevel.DENIED,
                reason=str(exc),
                digest=digest,
                version=spec.version,
                error=str(exc),
                error_type="ToolValidationError",
            )

        # The policy is consulted on every execution, never cached: a tool can
        # be revoked or downgraded between two calls.
        decision = self._policy.evaluate(spec)
        if not decision.allowed:
            if decision.requires_approval:
                valid, reason = self._policy.verify(
                    approval, tool=spec.name, version=spec.version, digest=digest
                )
                if not valid:
                    return self._finish(
                        execution_id=execution_id,
                        request=request,
                        started=started,
                        status=ToolStatus.APPROVAL_REQUIRED,
                        permission=PermissionLevel.APPROVAL,
                        reason=reason,
                        digest=digest,
                        version=spec.version,
                    )
                self._policy.consume(approval)
            else:
                return self._finish(
                    execution_id=execution_id,
                    request=request,
                    started=started,
                    status=ToolStatus.DENIED,
                    permission=decision.level,
                    reason=decision.reason,
                    digest=digest,
                    version=spec.version,
                )

        limit = self._default_timeout if timeout is None else float(timeout)
        deadline = started + limit if limit and limit > 0 else None
        context = ToolContext(
            execution_id=execution_id,
            caller=request.caller,
            deadline=deadline,
            now=self._clock,
        )

        self._registry.mark_in_use(spec.name)
        try:
            output, error, error_type = self._run(tool.handler, arguments, context, limit)
        finally:
            self._registry.release_in_use(spec.name)

        if error_type == "timeout":
            status = ToolStatus.TIMEOUT
        elif error is not None:
            status = ToolStatus.FAILED
        else:
            status = ToolStatus.SUCCESS

        return self._finish(
            execution_id=execution_id,
            request=request,
            started=started,
            status=status,
            permission=decision.level,
            reason=decision.reason if status is ToolStatus.SUCCESS else (error or ""),
            digest=digest,
            version=spec.version,
            output=output,
            error=error,
            error_type=error_type,
        )

    def _run(
        self,
        handler: Callable[[Any, ToolContext], Any],
        arguments: dict[str, Any],
        context: ToolContext,
        limit: float,
    ) -> tuple[Any, str | None, str | None]:
        """Run the handler, enforcing the deadline if one is set."""
        if not limit or limit <= 0:
            return self._call(handler, arguments, context)

        box: dict[str, Any] = {}

        def target() -> None:
            value, error, error_type = self._call(handler, arguments, context)
            box["value"] = value
            box["error"] = error
            box["error_type"] = error_type

        thread = threading.Thread(
            target=target,
            name=f"jarvis-tool-{context.execution_id}",
            daemon=True,
        )
        thread.start()
        thread.join(limit)
        if thread.is_alive():
            message = (
                f"execution exceeded {limit:g}s; the handler is still running "
                "in the background and its result is discarded"
            )
            return None, message, "timeout"
        return box.get("value"), box.get("error"), box.get("error_type")

    @staticmethod
    def _call(
        handler: Callable[[Any, ToolContext], Any],
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> tuple[Any, str | None, str | None]:
        try:
            return handler(arguments, context), None, None
        except Exception as exc:  # noqa: BLE001 - a tool must never crash JARVIS
            logger.warning(
                "tool %s raised %s", context.execution_id, type(exc).__name__, exc_info=exc
            )
            return None, scrub(str(exc)) or type(exc).__name__, type(exc).__name__

    def _finish(
        self,
        *,
        execution_id: str,
        request: ToolRequest,
        started: float,
        status: ToolStatus,
        permission: PermissionLevel,
        reason: str,
        digest: str,
        version: str,
        output: Any = None,
        error: str | None = None,
        error_type: str | None = None,
    ) -> ToolResult:
        duration_ms = (self._clock() - started) * 1000.0
        result = ToolResult(
            execution_id=execution_id,
            tool=request.tool,
            status=status,
            output=output,
            error=scrub(error) if error else None,
            error_type=error_type,
            permission=permission.value,
            reason=reason,
            duration_ms=duration_ms,
            request_id=request.request_id,
        )
        if self._audit is not None:
            self._audit.record(
                AuditRecord(
                    execution_id=execution_id,
                    tool=request.tool,
                    version=version,
                    timestamp=started,
                    status=status.value,
                    permission=permission.value,
                    reason=reason,
                    arguments_digest=digest,
                    argument_keys=request.argument_keys(),
                    duration_ms=duration_ms,
                    caller=request.caller,
                    error=result.error,
                )
            )
        logger.info(
            "tool %s -> %s (%s)", request.tool, status.value, reason or "no reason"
        )
        return result


def describe_decision(decision: PermissionDecision) -> str:
    """Short human-readable summary of a decision."""
    return f"{decision.level.value}: {decision.reason}"
