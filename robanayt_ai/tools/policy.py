"""Centralised permission policy (Phase 4).

Rules enforced here, in this order:

1. An explicit policy entry always wins over the tool's own declaration.
2. A tool declared `SAFE` (no side effects) may run without confirmation -
   but only while it is still declared safe.
3. Everything else is denied unless the policy knows it: a tool that the
   policy does not list is "unknown" and unknown tools never run.
4. A tool requiring approval runs only with a valid `ApprovalToken`, and that
   token is bound to the tool name, its version and the digest of the exact
   arguments. Change the arguments and the approval no longer applies.

A tool description, a model answer, a web page or a memory snippet is never
treated as permission: decisions come from this policy alone.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping

from robanayt_ai.tools.models import (
    ApprovalToken,
    PermissionDecision,
    PermissionLevel,
    ToolSpec,
    new_id,
)

logger = logging.getLogger(__name__)

DEFAULT_APPROVAL_TTL = 300.0

# Used approvals are remembered so that a token cannot be replayed. The record
# is bounded: the oldest id is dropped once the limit is reached, so a
# long-lived process cannot grow without limit. The TTL remains the primary
# guard - an evicted id is only reusable while its token has not expired.
MAX_CONSUMED_APPROVALS = 1024


class PermissionPolicy:
    """Decides whether a tool may run, and issues approvals."""

    def __init__(
        self,
        known: Mapping[str, PermissionLevel | str] | None = None,
        *,
        allow_declared_safe: bool = True,
        approval_ttl_seconds: float = DEFAULT_APPROVAL_TTL,
        clock: Callable[[], float] = time.time,
        max_consumed: int = MAX_CONSUMED_APPROVALS,
    ) -> None:
        self._known: dict[str, PermissionLevel] = {}
        for name, level in (known or {}).items():
            self._known[str(name)] = PermissionLevel(level)
        self._allow_declared_safe = allow_declared_safe
        self._ttl = float(approval_ttl_seconds)
        self._clock = clock
        self._max_consumed = max(0, int(max_consumed))
        # Insertion-ordered: the oldest id is evicted when the bound is hit.
        self._consumed: OrderedDict[str, None] = OrderedDict()

    # -- decisions --------------------------------------------------------

    def evaluate(self, spec: ToolSpec) -> PermissionDecision:
        """Check one tool right before execution. Never raises."""
        level = self.level_for(spec)
        if level is PermissionLevel.DENIED:
            return PermissionDecision(
                level=level,
                allowed=False,
                requires_approval=False,
                reason=self._deny_reason(spec),
            )
        if level is PermissionLevel.SAFE:
            return PermissionDecision(
                level=level,
                allowed=True,
                requires_approval=False,
                reason="declared safe: no external side effects",
            )
        return PermissionDecision(
            level=PermissionLevel.APPROVAL,
            allowed=False,
            requires_approval=True,
            reason="requires an approval bound to this action and its arguments",
        )

    def level_for(self, spec: ToolSpec) -> PermissionLevel:
        """Effective level: policy entry, then declaration, else deny."""
        override = self._known.get(spec.name)
        if override is not None:
            return override
        if spec.permission is PermissionLevel.SAFE and self._allow_declared_safe:
            return PermissionLevel.SAFE
        if spec.permission is PermissionLevel.DENIED:
            return PermissionLevel.DENIED
        return PermissionLevel.DENIED

    def _deny_reason(self, spec: ToolSpec) -> str:
        if spec.permission is PermissionLevel.DENIED:
            return "the tool declares itself as forbidden"
        return (
            f"tool {spec.name!r} is not covered by the policy; "
            "unknown tools are denied by default"
        )

    # -- approvals --------------------------------------------------------

    def grant(
        self,
        spec_or_name: ToolSpec | str,
        arguments_digest: str = "",
        *,
        version: str = "1.0.0",
        ttl_seconds: float | None = None,
    ) -> ApprovalToken:
        """Issue an approval for one exact action. Does not execute it."""
        if isinstance(spec_or_name, ToolSpec):
            tool = spec_or_name.name
            version = spec_or_name.version
        else:
            tool = str(spec_or_name)
        now = self._clock()
        ttl = self._ttl if ttl_seconds is None else float(ttl_seconds)
        token = ApprovalToken(
            token_id=new_id("apr"),
            tool=tool,
            version=version,
            arguments_digest=arguments_digest,
            issued_at=now,
            expires_at=now + max(ttl, 0.0),
        )
        logger.debug("approval granted (%s, digest=%s)", tool, arguments_digest)
        return token

    def verify(
        self,
        token: ApprovalToken | None,
        *,
        tool: str,
        version: str,
        digest: str,
    ) -> tuple[bool, str]:
        """Validate a token against the action about to be executed."""
        if token is None:
            return False, "no approval was provided"
        if token.token_id in self._consumed:
            return False, "this approval has already been used"
        return token.matches(tool=tool, version=version, digest=digest, now=self._clock())

    def consume(self, token: ApprovalToken) -> None:
        """Mark an approval as used - one approval, one execution."""
        self._remember(token)

    def revoke(self, token: ApprovalToken) -> None:
        """Withdraw an approval before it is used."""
        self._remember(token)

    def _remember(self, token: ApprovalToken) -> None:
        self._consumed[token.token_id] = None
        while len(self._consumed) > self._max_consumed:
            self._consumed.popitem(last=False)

    # -- introspection ----------------------------------------------------

    def known(self) -> dict[str, str]:
        return {name: level.value for name, level in sorted(self._known.items())}

    def describe(self) -> dict[str, object]:
        return {
            "known": self.known(),
            "allow_declared_safe": self._allow_declared_safe,
            "approval_ttl_seconds": self._ttl,
            "deny_unknown_by_default": True,
        }
