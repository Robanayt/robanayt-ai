"""Provider router: deterministic selection, no network, graceful failure."""

from __future__ import annotations

import pytest
from tests.ai.fakes import FakeProvider

from robanayt_ai.ai.errors import ProviderConfigurationError, ProviderUnavailableError
from robanayt_ai.ai.models import AIRequest, AIMessage
from robanayt_ai.ai.router import ProviderRouter, RoutingPolicy


def _request(metadata: dict | None = None) -> AIRequest:
    return AIRequest(messages=[AIMessage.user("hi")], metadata=metadata or {})


def test_policy_parsing() -> None:
    assert RoutingPolicy.from_value("local_first") is RoutingPolicy.LOCAL_FIRST
    assert RoutingPolicy.from_value("CLOUD_FIRST") is RoutingPolicy.CLOUD_FIRST
    with pytest.raises(ValueError, match="unknown routing policy"):
        RoutingPolicy.from_value("random")


def test_policy_ordering() -> None:
    assert RoutingPolicy.LOCAL_FIRST.order == ("local", "remote")
    assert RoutingPolicy.CLOUD_FIRST.order == ("remote", "local")


def test_policies_declare_what_they_really_do() -> None:
    """Phase 7: cost and quality policies are real; latency stays honest."""
    assert RoutingPolicy.LOCAL_FIRST.implemented is True
    assert RoutingPolicy.CHEAPEST.implemented is True
    assert RoutingPolicy.STRONGEST.implemented is True
    # Nothing measures latency: without declared hints the policy says so.
    assert RoutingPolicy.FASTEST.implemented is False
    assert "without hints it degrades to local_first" in RoutingPolicy.FASTEST.note
    assert "local first" in RoutingPolicy.FALLBACK.note
    assert "failover chain" in RoutingPolicy.FALLBACK.note


def test_local_first_picks_local_provider() -> None:
    local = FakeProvider("ollama", kind="local")
    remote = FakeProvider("openai_compatible", kind="remote")
    router = ProviderRouter([local, remote], policy=RoutingPolicy.LOCAL_FIRST)

    decision = router.select(_request())

    assert decision.provider is local
    assert decision.explicit is False
    assert "local_first" in decision.reason


def test_cloud_first_picks_remote_provider() -> None:
    local = FakeProvider("ollama", kind="local")
    remote = FakeProvider("openai_compatible", kind="remote")
    router = ProviderRouter([local, remote], policy=RoutingPolicy.CLOUD_FIRST)

    assert router.select(_request()).provider is remote


def test_explicit_provider_argument_wins() -> None:
    local = FakeProvider("ollama", kind="local")
    remote = FakeProvider("openai_compatible", kind="remote")
    router = ProviderRouter([local, remote])

    decision = router.select(_request(), provider="openai_compatible")

    assert decision.provider is remote
    assert decision.explicit is True


def test_explicit_provider_from_request_metadata() -> None:
    local = FakeProvider("ollama", kind="local")
    remote = FakeProvider("openai_compatible", kind="remote")
    router = ProviderRouter([local, remote], policy=RoutingPolicy.LOCAL_FIRST)

    decision = router.select(_request({"provider": "openai_compatible"}))

    assert decision.provider is remote


def test_unknown_provider_name_fails_loudly() -> None:
    router = ProviderRouter([FakeProvider("ollama")])

    with pytest.raises(ProviderConfigurationError, match="unknown provider"):
        router.select(_request(), provider="nope")


def test_unconfigured_explicit_provider_fails_gracefully() -> None:
    offline = FakeProvider("ollama", configured=False, message="no model configured")
    router = ProviderRouter([offline])

    with pytest.raises(ProviderUnavailableError, match="not configured"):
        router.select(_request(), provider="ollama")


def test_no_configured_provider_fails_with_details() -> None:
    router = ProviderRouter(
        [
            FakeProvider("ollama", configured=False, message="no model"),
            FakeProvider("openai_compatible", kind="remote", configured=False, message="no key"),
        ]
    )

    with pytest.raises(ProviderUnavailableError) as info:
        router.select(_request())

    assert "no model" in str(info.value)
    assert "no key" in str(info.value)


def test_status_defaults_to_no_probe() -> None:
    provider = FakeProvider("ollama")
    router = ProviderRouter([provider])

    statuses = router.status()

    assert [s.provider for s in statuses] == ["ollama"]
    assert statuses[0].configured is True
    # FakeProvider never touches the network regardless of `probe`.


def test_describe_exposes_policy_and_providers() -> None:
    router = ProviderRouter([FakeProvider("ollama")], policy=RoutingPolicy.FALLBACK)

    summary = router.describe()

    assert summary["policy"] == "fallback"
    assert summary["providers"] == ["ollama"]
    assert "api_key" not in summary
