"""Provider router (Phase 7: hybrid routing).

Deterministic, explainable and network-free:

- An explicit provider name (argument or request metadata) is honoured
  strictly: if it is not usable, routing fails with a clear error instead of
  silently switching vendors.
- Otherwise a policy orders the configured providers: by kind (local/cloud
  first), by declared cost, by declared quality, or by declared latency.
- Privacy can force local-only routing; `allow_remote=False` removes remote
  providers from consideration entirely (the run-mode boundary).
- `status(probe=False)` and `select()`/`explain()` never touch the network, so
  routing works fully offline and every decision is reproducible in tests.

Nothing here measures anything: cost, quality and latency are *declared* by a
provider (see `jarvis.ai.provider.ProviderProfile`). When a provider declares
nothing, the router says so in the reason instead of inventing a number - no
fake intelligence is claimed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jarvis.ai.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
)
from jarvis.ai.models import AIRequest, ProviderStatus
from jarvis.ai.provider import LOCAL, REMOTE, Provider, ProviderProfile, profile_of

logger = logging.getLogger(__name__)

# Request-level privacy flag: when set, remote providers are excluded.
LOCAL_ONLY = "local_only"
PRIVACY_METADATA_KEY = "privacy"


class RoutingPolicy(str, Enum):
    """How a provider is chosen when none is named explicitly."""

    LOCAL_FIRST = "local_first"
    CLOUD_FIRST = "cloud_first"
    FALLBACK = "fallback"
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    STRONGEST = "strongest"

    @classmethod
    def from_value(cls, value: Any) -> RoutingPolicy:
        if isinstance(cls, value.__class__) and isinstance(value, cls):
            return value
        if isinstance(value, str):
            candidate = value.strip().lower()
            for policy in cls:
                if policy.value == candidate:
                    return policy
        options = ", ".join(policy.value for policy in cls)
        raise ValueError(f"unknown routing policy {value!r}; allowed: {options}")

    @property
    def order(self) -> tuple[str, ...]:
        """Provider kinds in the order they are considered."""
        if self is RoutingPolicy.CLOUD_FIRST:
            return (REMOTE, LOCAL)
        return (LOCAL, REMOTE)

    @property
    def implemented(self) -> bool:
        """False only for the policy that cannot be honest without data."""
        return self is not RoutingPolicy.FASTEST

    @property
    def note(self) -> str:
        """What the policy really does - no aspirational wording."""
        if self is RoutingPolicy.LOCAL_FIRST:
            return "local providers first, then remote (deterministic)"
        if self is RoutingPolicy.CLOUD_FIRST:
            return "remote providers first, then local (deterministic)"
        if self is RoutingPolicy.FALLBACK:
            return "local first; the remaining providers form a failover chain"
        if self is RoutingPolicy.CHEAPEST:
            return "orders by declared cost (0 = no per-request fee), local first on ties"
        if self is RoutingPolicy.STRONGEST:
            return (
                "orders by declared quality tier; providers that declare "
                "nothing rank last"
            )
        return (
            "orders by declared latency hints only; without hints it degrades "
            "to local_first and says so in the reason"
        )


@dataclass(frozen=True)
class RoutingDecision:
    """Why a provider was chosen - useful for logs and diagnostics.

    `chain` holds the ordered alternatives that a failover may use; each one
    is tried at most once. `rejections` explains every provider that was
    considered and skipped, so a refusal is never silent.
    """

    provider: Provider
    policy: RoutingPolicy
    explicit: bool
    reason: str
    candidates: tuple[str, ...] = ()
    chain: tuple[str, ...] = ()
    rejections: tuple[tuple[str, str], ...] = ()
    model: str | None = None
    privacy: str = LOCAL
    profile: ProviderProfile | None = None


class ProviderRouter:
    """Chooses a provider; never performs I/O of its own."""

    def __init__(
        self,
        providers: list[Provider] | tuple[Provider, ...],
        *,
        default_provider: str = "ollama",
        policy: RoutingPolicy = RoutingPolicy.LOCAL_FIRST,
        allow_remote: bool = True,
    ) -> None:
        self._providers: dict[str, Provider] = {}
        for provider in providers:
            self._providers[provider.name] = provider
        self.default_provider = default_provider or "ollama"
        self.policy = policy
        # The run-mode boundary: when remote calls are not allowed, remote
        # providers are not merely reported - they are never selected.
        self.allow_remote = bool(allow_remote)

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(self._providers)

    def get(self, name: str) -> Provider:
        try:
            return self._providers[name]
        except KeyError:
            raise ProviderConfigurationError(
                f"unknown provider {name!r}; known: {', '.join(self.provider_names)}"
            ) from None

    def register(self, provider: Provider) -> None:
        self._providers[provider.name] = provider

    def status(self, probe: bool = False) -> tuple[ProviderStatus, ...]:
        """Status of every provider; a probe touches the network only if asked."""
        return tuple(provider.status(probe=probe) for provider in self._providers.values())

    def select(
        self,
        request: AIRequest | None = None,
        *,
        provider: str | None = None,
        privacy: str | None = None,
    ) -> RoutingDecision:
        """Pick a provider. Raises when nothing usable is configured.

        `privacy=LOCAL_ONLY` (or `metadata["privacy"] = "local_only"`) excludes
        remote providers instead of sending data off the machine.
        """
        requested = provider or (request.requested_provider() if request else None)
        wanted = self._privacy(request, privacy)

        if requested:
            return self._select_explicit(requested, request, wanted)

        ordered = self._ordered()
        rejections: list[tuple[str, str]] = []
        usable: list[Provider] = []
        for candidate in ordered:
            state = candidate.status(probe=False)
            blocked = self._blocked(candidate, state, wanted)
            if blocked is None:
                usable.append(candidate)
            else:
                rejections.append((candidate.name, blocked))

        if not usable:
            details = "; ".join(f"{name}: {reason}" for name, reason in rejections)
            raise ProviderUnavailableError(
                f"no configured provider for policy {self.policy.value} ({details})"
            )

        primary = usable[0]
        return self._decision(
            primary,
            request,
            explicit=False,
            reason=self._reason(primary, ordered),
            candidates=tuple(item.name for item in ordered),
            chain=tuple(item.name for item in usable[1:]),
            rejections=tuple(rejections),
        )

    def explain(self, request: AIRequest | None = None) -> dict[str, Any]:
        """What would be used, and why everything else would not. No network."""
        summary: dict[str, Any] = {
            "policy": self.policy.value,
            "policy_note": self.policy.note,
            "allow_remote": self.allow_remote,
            "default_provider": self.default_provider,
            "providers": [
                {
                    "name": provider.name,
                    "kind": provider.kind,
                    "model": getattr(provider, "model", "") or None,
                    "profile": profile_of(provider).as_dict(),
                    "status": provider.status(probe=False).as_dict(),
                }
                for provider in self._providers.values()
            ],
        }
        try:
            decision = self.select(request)
        except (ProviderUnavailableError, ProviderConfigurationError) as exc:
            summary.update(
                {
                    "selected": None,
                    "reason": str(exc),
                    "chain": [],
                    "rejected": [
                        {"provider": name, "reason": reason}
                        for name, reason in self._rejections_all(
                            self._privacy(request, None)
                        )
                    ],
                }
            )
            return summary
        summary.update(
            {
                "selected": {
                    "provider": decision.provider.name,
                    "model": decision.model,
                    "kind": decision.provider.kind,
                    "privacy": decision.privacy,
                    "reason": decision.reason,
                },
                "chain": list(decision.chain),
                "rejected": [
                    {"provider": name, "reason": reason}
                    for name, reason in decision.rejections
                ],
            }
        )
        return summary

    def describe(self) -> dict[str, Any]:
        """Diagnostics-only summary (no credentials)."""
        return {
            "default_provider": self.default_provider,
            "policy": self.policy.value,
            "policy_note": self.policy.note,
            "allow_remote": self.allow_remote,
            "providers": [provider.name for provider in self._providers.values()],
        }

    # -- internals ---------------------------------------------------------

    def _select_explicit(
        self, requested: str, request: AIRequest | None, wanted: str | None
    ) -> RoutingDecision:
        """An explicit provider is honoured strictly - never substituted."""
        chosen = self.get(requested)
        state = chosen.status(probe=False)
        if not state.configured:
            raise ProviderUnavailableError(
                f"provider {requested!r} is not configured: {state.message}",
                provider=requested,
            )
        blocked = self._blocked(chosen, state, wanted)
        if blocked is not None:
            raise ProviderUnavailableError(
                f"provider {requested!r} is not usable: {blocked}", provider=requested
            )
        return self._decision(
            chosen,
            request,
            explicit=True,
            reason=f"explicitly requested provider {requested!r}",
        )

    def _decision(
        self,
        provider: Provider,
        request: AIRequest | None,
        *,
        explicit: bool,
        reason: str,
        candidates: tuple[str, ...] = (),
        chain: tuple[str, ...] = (),
        rejections: tuple[tuple[str, str], ...] = (),
    ) -> RoutingDecision:
        model = None
        if request is not None:
            model = request.model
        if not model:
            model = getattr(provider, "model", "") or None
        return RoutingDecision(
            provider=provider,
            policy=self.policy,
            explicit=explicit,
            reason=reason,
            candidates=candidates or self.provider_names,
            chain=chain,
            rejections=rejections,
            model=model,
            privacy=provider.kind,
            profile=profile_of(provider),
        )

    def _blocked(
        self, provider: Provider, state: ProviderStatus, wanted: str | None
    ) -> str | None:
        """Return why a provider cannot be used, or `None` when it can."""
        if not state.configured:
            return state.message or "not configured"
        # Boundaries first: they are decisions, not availability accidents.
        if provider.kind == REMOTE and not self.allow_remote:
            return "remote providers are disabled (allow_remote=false)"
        if wanted == LOCAL_ONLY and provider.kind == REMOTE:
            return "privacy=local_only excludes remote providers"
        if not state.available:
            return state.message or "not available"
        return None

    def _rejections_all(self, wanted: str | None = None) -> tuple[tuple[str, str], ...]:
        """Why every provider is unusable - used when selection fails."""
        rejections: list[tuple[str, str]] = []
        for provider in self._providers.values():
            state = provider.status(probe=False)
            blocked = self._blocked(provider, state, wanted)
            rejections.append(
                (provider.name, blocked or (state.message or "considered"))
            )
        return tuple(rejections)

    @staticmethod
    def _privacy(request: AIRequest | None, privacy: str | None) -> str | None:
        value = privacy
        if value is None and request is not None:
            raw = request.metadata.get(PRIVACY_METADATA_KEY)
            value = str(raw) if raw else None
        if not value:
            return None
        return value.strip().lower()

    def _ordered(self) -> list[Provider]:
        """Configured providers in policy order. Deterministic, no guessing."""
        providers = list(self._providers.values())
        policy = self.policy

        if policy is RoutingPolicy.CHEAPEST:
            return sorted(
                providers,
                key=lambda item: (
                    profile_of(item).cost,
                    _kind_rank(item),
                    _index(item, providers),
                ),
            )
        if policy is RoutingPolicy.STRONGEST:
            return sorted(
                providers,
                key=lambda item: (
                    _quality_rank(item),
                    _tools_rank(item),
                    _index(item, providers),
                ),
            )
        if policy is RoutingPolicy.FASTEST:
            return sorted(
                providers,
                key=lambda item: (
                    _latency_rank(item),
                    _kind_rank(item),
                    _index(item, providers),
                ),
            )
        # Kind-based policies (local_first, cloud_first, fallback).
        first = policy.order[0]
        return sorted(
            providers, key=lambda item: (0 if item.kind == first else 1, _index(item, providers))
        )

    def _reason(self, provider: Provider, ordered: list[Provider]) -> str:
        policy = self.policy
        profile = profile_of(provider)
        name = provider.name

        if policy is RoutingPolicy.CHEAPEST:
            declared = [item for item in ordered if profile_of(item).cost == profile.cost]
            tie = f" (tie with {len(declared) - 1} other(s); local first)" if len(declared) > 1 else ""
            return f"policy cheapest selected {name!r} (declared cost {profile.cost}){tie}"
        if policy is RoutingPolicy.STRONGEST:
            if profile.quality is None:
                others = [item for item in ordered if profile_of(item).quality is not None]
                note = (
                    "no provider declares a quality tier; registration order used"
                    if not others
                    else f"{name!r} declares no quality tier; ranked after declared ones"
                )
                return f"policy strongest selected {name!r}: {note}"
            return f"policy strongest selected {name!r} (declared quality {profile.quality})"
        if policy is RoutingPolicy.FASTEST:
            if profile.latency_hint_ms is None:
                return (
                    f"policy fastest selected {name!r}: no provider declares a "
                    "latency hint; degraded to local_first ordering (no measurement)"
                )
            return (
                f"policy fastest selected {name!r} "
                f"(declared latency {profile.latency_hint_ms} ms)"
            )
        if policy is RoutingPolicy.FALLBACK:
            return f"policy fallback selected {name!r} ({provider.kind}); failover chain follows"
        return f"policy {policy.value} selected {policy.order[0]} provider {name!r}"


def _index(provider: Provider, providers: list[Provider]) -> int:
    try:
        return providers.index(provider)
    except ValueError:  # pragma: no cover - defensive
        return len(providers)


def _kind_rank(provider: Provider) -> int:
    return 0 if provider.kind == LOCAL else 1


def _quality_rank(provider: Provider) -> tuple[int, int]:
    """Declared quality first (higher is stronger); unknown quality last."""
    quality = profile_of(provider).quality
    if quality is None:
        return (1, 0)
    return (0, -int(quality))


def _tools_rank(provider: Provider) -> int:
    capabilities = getattr(provider, "capabilities", None)
    return 0 if getattr(capabilities, "tools", False) else 1


def _latency_rank(provider: Provider) -> tuple[int, int]:
    """Declared latency ascending; providers without a hint rank last."""
    hint = profile_of(provider).latency_hint_ms
    if hint is None:
        return (1, 0)
    return (0, int(hint))
