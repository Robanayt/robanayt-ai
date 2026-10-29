"""Phase 7 - hybrid routing: cost, quality, latency, privacy and failover.

Every test is offline and deterministic: nothing here contacts a provider.
"""

from __future__ import annotations

import pytest
from tests.ai.fakes import FakeProvider

from robanayt_ai.ai.errors import ProviderAuthenticationError, ProviderUnavailableError
from robanayt_ai.ai.models import AIRequest, AIMessage
from robanayt_ai.ai.provider import ProviderProfile
from robanayt_ai.ai.router import LOCAL_ONLY, ProviderRouter, RoutingPolicy
from robanayt_ai.ai.service import AIService


def _request(metadata: dict | None = None, model: str | None = None) -> AIRequest:
    return AIRequest(
        messages=[AIMessage.user("hi")], metadata=metadata or {}, model=model
    )


def _service(router: ProviderRouter, **kwargs: object) -> AIService:
    return AIService(router, default_model="test-model", **kwargs)  # type: ignore[arg-type]


# -- cheapest --------------------------------------------------------------


def test_cheapest_prefers_the_cheaper_provider() -> None:
    local = FakeProvider("ollama", profile=ProviderProfile(cost=3))
    remote = FakeProvider("cloud", kind="remote", profile=ProviderProfile(cost=1))
    router = ProviderRouter([local, remote], policy=RoutingPolicy.CHEAPEST)

    decision = router.select(_request())

    assert decision.provider is remote
    assert "cost 1" in decision.reason


def test_cheapest_breaks_ties_towards_local() -> None:
    remote = FakeProvider("cloud", kind="remote", profile=ProviderProfile(cost=0))
    local = FakeProvider("ollama", profile=ProviderProfile(cost=0))
    router = ProviderRouter([remote, local], policy=RoutingPolicy.CHEAPEST)

    assert router.select(_request()).provider is local


def test_cheapest_reports_when_cost_is_a_tie() -> None:
    first = FakeProvider("a", profile=ProviderProfile(cost=1))
    second = FakeProvider("b", profile=ProviderProfile(cost=1))
    router = ProviderRouter([first, second], policy=RoutingPolicy.CHEAPEST)

    assert "tie" in router.select(_request()).reason


# -- strongest -------------------------------------------------------------


def test_strongest_prefers_the_declared_quality_tier() -> None:
    weak = FakeProvider("weak", profile=ProviderProfile(quality=1))
    strong = FakeProvider("strong", profile=ProviderProfile(quality=3))
    router = ProviderRouter([weak, strong], policy=RoutingPolicy.STRONGEST)

    decision = router.select(_request())

    assert decision.provider is strong
    assert "quality 3" in decision.reason


def test_strongest_ranks_unknown_quality_last() -> None:
    unknown = FakeProvider("unknown", profile=ProviderProfile(quality=None))
    declared = FakeProvider("declared", profile=ProviderProfile(quality=0))
    router = ProviderRouter([unknown, declared], policy=RoutingPolicy.STRONGEST)

    decision = router.select(_request())

    assert decision.provider is declared
    assert "unknown" not in decision.reason  # the reason names no fake winner


def test_strongest_says_so_when_nobody_declares_quality() -> None:
    first = FakeProvider("first")
    router = ProviderRouter([first], policy=RoutingPolicy.STRONGEST)

    assert "no provider declares a quality tier" in router.select(_request()).reason


# -- fastest (honest about missing data) -----------------------------------


def test_fastest_uses_declared_latency_hints() -> None:
    slow = FakeProvider("slow", profile=ProviderProfile(latency_hint_ms=900))
    fast = FakeProvider("fast", kind="remote", profile=ProviderProfile(latency_hint_ms=80))
    router = ProviderRouter([slow, fast], policy=RoutingPolicy.FASTEST)

    decision = router.select(_request())

    assert decision.provider is fast
    assert "80 ms" in decision.reason


def test_fastest_without_hints_degrades_to_local_first_and_says_so() -> None:
    local = FakeProvider("ollama")
    remote = FakeProvider("cloud", kind="remote")
    router = ProviderRouter([remote, local], policy=RoutingPolicy.FASTEST)

    decision = router.select(_request())

    assert decision.provider is local
    assert "no provider declares a latency hint" in decision.reason
    assert "no measurement" in decision.reason


# -- privacy ---------------------------------------------------------------


def test_local_only_privacy_excludes_remote_providers() -> None:
    local = FakeProvider("ollama")
    remote = FakeProvider("cloud", kind="remote")
    router = ProviderRouter([remote, local], policy=RoutingPolicy.CLOUD_FIRST)

    decision = router.select(_request(), privacy=LOCAL_ONLY)

    assert decision.provider is local
    assert ("cloud", "privacy=local_only excludes remote providers") in decision.rejections


def test_privacy_can_come_from_request_metadata() -> None:
    local = FakeProvider("ollama")
    remote = FakeProvider("cloud", kind="remote")
    router = ProviderRouter([remote, local], policy=RoutingPolicy.CLOUD_FIRST)

    decision = router.select(_request({"privacy": "local_only"}))

    assert decision.provider is local


def test_local_only_without_a_local_provider_fails_loudly() -> None:
    router = ProviderRouter([FakeProvider("cloud", kind="remote")])

    with pytest.raises(ProviderUnavailableError, match="local_only"):
        router.select(_request(), privacy=LOCAL_ONLY)


def test_explicit_remote_provider_is_refused_under_local_only() -> None:
    """Privacy wins over an explicit choice: no silent downgrade."""
    router = ProviderRouter([FakeProvider("cloud", kind="remote")])

    with pytest.raises(ProviderUnavailableError, match="local_only"):
        router.select(_request(), provider="cloud", privacy=LOCAL_ONLY)


# -- the remote boundary ---------------------------------------------------


def test_blocked_remote_provider_is_never_selected() -> None:
    """Regression: a remote provider was selectable while remote calls were off."""
    remote = FakeProvider("cloud", kind="remote", configured=True, available=False)
    router = ProviderRouter([remote], allow_remote=False)

    with pytest.raises(ProviderUnavailableError, match="allow_remote=false"):
        router.select(_request())


def test_remote_boundary_is_reported_as_a_rejection() -> None:
    local = FakeProvider("ollama")
    remote = FakeProvider("cloud", kind="remote")
    router = ProviderRouter([local, remote], allow_remote=False)

    decision = router.select(_request())

    assert decision.provider is local
    assert ("cloud", "remote providers are disabled (allow_remote=false)") in decision.rejections


def test_explicit_remote_provider_is_refused_when_remote_is_blocked() -> None:
    router = ProviderRouter([FakeProvider("cloud", kind="remote")], allow_remote=False)

    with pytest.raises(ProviderUnavailableError, match="allow_remote=false"):
        router.select(_request(), provider="cloud")


# -- decision contents -----------------------------------------------------


def test_decision_exposes_model_chain_and_rejections() -> None:
    local = FakeProvider("ollama", model="llama3")
    remote = FakeProvider("cloud", kind="remote", model="gpt-x")
    router = ProviderRouter([local, remote], policy=RoutingPolicy.LOCAL_FIRST)

    decision = router.select(_request(model="asked-model"))

    assert decision.model == "asked-model"
    assert decision.chain == ("cloud",)
    assert decision.privacy == "local"
    assert decision.rejections == ()


def test_decision_falls_back_to_the_provider_model() -> None:
    local = FakeProvider("ollama", model="llama3")
    router = ProviderRouter([local])

    assert router.select(_request()).model == "llama3"


def test_explain_shows_the_active_choice_and_the_reasons_against_others() -> None:
    local = FakeProvider("ollama", model="llama3")
    remote = FakeProvider("cloud", kind="remote", configured=False, message="no key")
    router = ProviderRouter([local, remote], allow_remote=False)

    explanation = router.explain(_request())

    assert explanation["selected"]["provider"] == "ollama"
    assert explanation["selected"]["model"] == "llama3"
    assert explanation["allow_remote"] is False
    rejected = {item["provider"]: item["reason"] for item in explanation["rejected"]}
    assert "no key" in rejected["cloud"] or "disabled" in rejected["cloud"]


def test_explain_reports_why_nothing_can_be_used() -> None:
    router = ProviderRouter([FakeProvider("ollama", configured=False, message="no model")])

    explanation = router.explain(_request())

    assert explanation["selected"] is None
    assert "no model" in explanation["reason"]
    assert explanation["rejected"][0]["provider"] == "ollama"


# -- failover --------------------------------------------------------------


def test_failover_uses_the_next_provider_and_records_it() -> None:
    primary = FakeProvider("ollama", failures=[ProviderUnavailableError("down", provider="ollama")])
    secondary = FakeProvider("cloud", kind="remote", reply="from cloud")
    router = ProviderRouter([primary, secondary], policy=RoutingPolicy.LOCAL_FIRST)
    service = _service(router, allow_fallback=True)

    response = service.generate(_request())

    assert response.content == "from cloud"
    assert response.metadata["routing_failover"]["from"] == "ollama"
    assert response.metadata["routing_failover"]["to"] == "cloud"


def test_failover_is_off_by_default() -> None:
    primary = FakeProvider("ollama", failures=[ProviderUnavailableError("down", provider="ollama")])
    router = ProviderRouter([primary, FakeProvider("cloud", kind="remote")])
    service = _service(router)

    with pytest.raises(ProviderUnavailableError):
        service.generate(_request())


def test_failover_never_applies_to_non_retryable_errors() -> None:
    primary = FakeProvider(
        "ollama", failures=[ProviderAuthenticationError("bad key", provider="ollama")]
    )
    secondary = FakeProvider("cloud", kind="remote")
    router = ProviderRouter([primary, secondary])
    service = _service(router, allow_fallback=True)

    with pytest.raises(ProviderAuthenticationError):
        service.generate(_request())

    assert secondary.calls == []  # an auth failure is final, not re-routed


def test_failover_is_bounded_by_the_chain() -> None:
    """Every alternative is tried once - no loop, no endless retry."""
    error = ProviderUnavailableError("down", provider="x")
    primary = FakeProvider("a", failures=[error])
    second = FakeProvider("b", failures=[error])
    third = FakeProvider("c", failures=[error])
    router = ProviderRouter([primary, second, third])
    service = _service(router, allow_fallback=True)

    with pytest.raises(ProviderUnavailableError):
        service.generate(_request())

    assert len(primary.calls) == 1
    assert len(second.calls) == 1
    assert len(third.calls) == 1


def test_streaming_is_never_rerouted_mid_flight() -> None:
    primary = FakeProvider("ollama", failures=[ProviderUnavailableError("down", provider="ollama")])
    router = ProviderRouter([primary, FakeProvider("cloud", kind="remote")])
    service = _service(router, allow_fallback=True)

    with pytest.raises(ProviderUnavailableError):
        list(service.stream(_request()))


# -- service surface -------------------------------------------------------


def test_service_explain_reports_policy_and_active_model() -> None:
    local = FakeProvider("ollama", model="llama3")
    router = ProviderRouter([local], policy=RoutingPolicy.CHEAPEST)
    service = _service(router)

    explanation = service.explain(_request())

    assert explanation["policy"] == "cheapest"
    assert explanation["selected"]["model"] == "llama3"
    assert explanation["default_model"] == "test-model"


def test_service_explain_marks_a_disabled_layer() -> None:
    router = ProviderRouter([FakeProvider("ollama")])
    service = _service(router, enabled=False)

    explanation = service.explain(_request())

    assert "disabled" in explanation["reason"]


def test_describe_includes_routing_without_credentials() -> None:
    router = ProviderRouter([FakeProvider("ollama")])
    service = _service(router)

    summary = service.describe()
    blob = str(summary).lower()

    assert "routing" in summary
    assert "api_key" not in blob
    assert "sk-" not in blob
