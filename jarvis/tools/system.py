"""Composition root of the tool subsystem (Phase 4).

One factory builds the whole thing so that the application, the tests and
future callers all get the same wiring. Building it performs no I/O: no
network, no database, no tool execution - only registration.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.core.bus import EventBus
from jarvis.core.config import ToolsConfig
from jarvis.tools import builtins
from jarvis.tools.audit import AuditLog
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.policy import PermissionPolicy
from jarvis.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolSystem:
    """Registry, policy, audit log and executor wired together."""

    registry: ToolRegistry
    policy: PermissionPolicy
    audit: AuditLog
    executor: ToolExecutor

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.executor.enabled,
            "tools": list(self.registry.names()),
            "policy": self.policy.describe(),
            "default_timeout_seconds": self.executor.default_timeout,
        }

    def close(self) -> None:
        """Explicit shutdown hook (nothing to release today)."""
        return None


def build_tool_system(
    config: ToolsConfig | None = None,
    *,
    bus: EventBus | None = None,
    clock: Callable[[], float] = time.time,
    with_builtins: bool = True,
) -> ToolSystem:
    """Build the tool subsystem. Never contacts anything or runs a tool."""
    config = config or ToolsConfig()
    registry = ToolRegistry()
    if with_builtins:
        builtins.register_builtins(registry)
    policy = builtins.builtin_policy(
        approval_ttl_seconds=config.approval_ttl_seconds, clock=clock
    )
    audit_path: Path | None = None
    if config.audit_to_file:
        audit_path = Path(config.audit_directory) / config.audit_filename
    audit = AuditLog(
        bus=bus, path=audit_path, clock=clock, enabled=config.audit_enabled
    )
    executor = ToolExecutor(
        registry,
        policy,
        audit=audit,
        default_timeout=config.default_timeout_seconds,
        clock=clock,
        enabled=config.enabled,
    )
    return ToolSystem(registry=registry, policy=policy, audit=audit, executor=executor)
