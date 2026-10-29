"""The provider contract.

Everything above this layer (service, router, future dialogue/memory/agents)
depends only on this interface - never on Ollama or OpenAI specifics, and
never on how HTTP is performed.

A provider must be constructible without touching the network: construction
may only read configuration. `status(probe=True)` is the single place where
an actual connection may happen, and it is opt-in.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.ai.models import AIChunk, AIRequest, AIResponse, ProviderStatus

LOCAL = "local"
REMOTE = "remote"


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can do. Router and service never assume more."""

    streaming: bool = False
    local: bool = False
    tools: bool = False


@dataclass(frozen=True)
class ProviderProfile:
    """Declared routing metadata: cost, quality tier and latency hint.

    Everything here is a *declaration*, not a measurement:

    - `cost`     - relative per-request cost, 0 = no per-request fee
                   (typical for local inference), higher = more expensive.
    - `quality`  - 0..3 declared capability tier; `None` means "unknown" and
                   never wins a comparison against a declared value.
    - `latency_hint_ms` - declared typical latency; `None` means "no data".
                   Routing then says so instead of inventing a number.

    A provider may declare a profile (`provider.profile`); when it does not,
    the router derives a conservative default from its kind.
    """

    cost: int = 0
    quality: int | None = None
    latency_hint_ms: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost", max(0, int(self.cost)))
        if self.quality is not None:
            object.__setattr__(self, "quality", max(0, int(self.quality)))
        if self.latency_hint_ms is not None:
            object.__setattr__(self, "latency_hint_ms", max(0, int(self.latency_hint_ms)))
        object.__setattr__(self, "notes", str(self.notes or ""))

    def as_dict(self) -> dict[str, object]:
        return {
            "cost": self.cost,
            "quality": self.quality,
            "latency_hint_ms": self.latency_hint_ms,
            "notes": self.notes,
        }


# Conservative defaults for providers that declare nothing of their own.
LOCAL_PROFILE = ProviderProfile(cost=0, notes="local inference: no per-request fee")
REMOTE_PROFILE = ProviderProfile(cost=2, notes="remote API: billed per request")


def profile_of(provider: object) -> ProviderProfile:
    """Return the profile a provider declares, or a default by its kind."""
    profile = getattr(provider, "profile", None)
    if isinstance(profile, ProviderProfile):
        return profile
    kind = str(getattr(provider, "kind", "") or "")
    return LOCAL_PROFILE if kind == LOCAL else REMOTE_PROFILE


@runtime_checkable
class Provider(Protocol):
    """Structural contract every provider must satisfy."""

    name: str
    kind: str
    capabilities: ProviderCapabilities

    def status(self, probe: bool = False) -> ProviderStatus:
        """Configuration state; probes the network only when `probe=True`."""
        ...

    def generate(self, request: AIRequest) -> AIResponse: ...

    def stream(self, request: AIRequest) -> Iterator[AIChunk]: ...


def supports_streaming(provider: Provider) -> bool:
    """True when the provider declares streaming support."""
    capabilities = getattr(provider, "capabilities", None)
    return bool(capabilities and capabilities.streaming)


def supports_tools(provider: Provider) -> bool:
    """True when the provider can return native structured tool calls."""
    capabilities = getattr(provider, "capabilities", None)
    return bool(capabilities and capabilities.tools)
