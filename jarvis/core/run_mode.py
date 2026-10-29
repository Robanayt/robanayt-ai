"""Run modes: what the application is allowed to do in this process.

Kept intentionally small - four modes with a policy each. The policy is a
plain frozen dataclass so it can be logged, tested and compared without
touching any runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RunMode(str, Enum):
    """How JARVIS_AI was started."""

    NORMAL = "normal"
    SAFE = "safe"
    DIAGNOSTIC = "diagnostic"
    DEVELOPMENT = "development"

    @classmethod
    def from_value(cls, value: Any) -> RunMode:
        """Resolve a mode from a string/RunMode; `None` means NORMAL."""
        if value is None:
            return cls.NORMAL
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            candidate = value.strip().lower()
            for mode in cls:
                if mode.value == candidate:
                    return mode
        options = ", ".join(mode.value for mode in cls)
        raise ValueError(f"unknown run mode {value!r}; allowed: {options}")

    @property
    def policy(self) -> RunModePolicy:
        return _POLICIES[self]

    def describe(self) -> dict[str, Any]:
        policy = self.policy
        return {
            "run_mode": self.value,
            "allow_network": policy.allow_network,
            "allow_tools": policy.allow_tools,
            "mock_backends": policy.mock_backends,
            "log_level": policy.log_level,
        }


@dataclass(frozen=True)
class RunModePolicy:
    """What a mode permits. `allow_*` flags are capability boundaries."""

    allow_network: bool
    allow_tools: bool
    mock_backends: bool
    log_level: str


_POLICIES: dict[RunMode, RunModePolicy] = {
    # Everyday use: tools allowed, nothing leaves the machine.
    RunMode.NORMAL: RunModePolicy(
        allow_network=False, allow_tools=True, mock_backends=False, log_level="INFO"
    ),
    # Restricted: no tools, no side effects - conversation and diagnostics only.
    RunMode.SAFE: RunModePolicy(
        allow_network=False, allow_tools=False, mock_backends=False, log_level="INFO"
    ),
    # Investigating the system: verbose logs, backends may be stubbed.
    RunMode.DIAGNOSTIC: RunModePolicy(
        allow_network=False, allow_tools=False, mock_backends=True, log_level="DEBUG"
    ),
    # Local development: verbose logs plus stubs; still no implicit network.
    RunMode.DEVELOPMENT: RunModePolicy(
        allow_network=False, allow_tools=True, mock_backends=True, log_level="DEBUG"
    ),
}
