"""Phase 7 in the real pipeline: a dialogue turn routed by policy.

The fake providers answer locally - no HTTP, no network, no credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.ai.errors import ProviderUnavailableError
from jarvis.ai.provider import ProviderProfile
from jarvis.ai.router import ProviderRouter, RoutingPolicy
from jarvis.ai.service import AIService
from jarvis.core.config import OrchestrationConfig, ToolsConfig
from jarvis.orchestration import DialogService
from jarvis.orchestration.models import DialogStatus
from jarvis.tools import build_tool_system
from tests.orchestration.dialog_fakes import (
    FakeClock,
    ScriptedProvider,
    answer,
    memory_service,
    structured_call,
)


def _dialog(
    tmp_path: Path,
    *,
    policy: RoutingPolicy = RoutingPolicy.LOCAL_FIRST,
    local_cost: int = 0,
    remote_cost: int = 2,
    local_reply: object | None = None,
    remote_reply: object | None = None,
    allow_fallback: bool = False,
) -> tuple[DialogService, ScriptedProvider, ScriptedProvider]:
    clock = FakeClock()
    memory = memory_service(tmp_path)
    tools = build_tool_system(ToolsConfig(), clock=clock)

    local = ScriptedProvider([local_reply or answer("from local")], name="local_ai")
    local.profile = ProviderProfile(cost=local_cost)
    remote = ScriptedProvider([remote_reply or answer("from cloud")], name="cloud_ai")
    # The fake declares its kind explicitly, as a real provider does.
    remote.kind = "remote"
    remote.profile = ProviderProfile(cost=remote_cost)

    router = ProviderRouter([local, remote], default_provider=local.name, policy=policy)
    service = AIService(
        router, default_model="test-model", allow_fallback=allow_fallback
    )
    dialog = DialogService(service, memory, tools, OrchestrationConfig(), clock=clock)
    return dialog, local, remote


def test_cheapest_policy_routes_the_turn_to_the_cheaper_provider(tmp_path: Path) -> None:
    dialog, local, remote = _dialog(
        tmp_path, policy=RoutingPolicy.CHEAPEST, local_cost=3, remote_cost=1
    )

    result = dialog.send("привет")

    assert result.ok
    assert result.answer == "from cloud"
    assert len(remote.requests) == 1
    assert local.requests == []


def test_local_first_keeps_the_turn_local_even_when_it_costs_more(tmp_path: Path) -> None:
    dialog, local, remote = _dialog(
        tmp_path, policy=RoutingPolicy.LOCAL_FIRST, local_cost=3, remote_cost=1
    )

    result = dialog.send("привет")

    assert result.ok
    assert result.answer == "from local"
    assert len(local.requests) == 1
    assert remote.requests == []


def test_local_only_privacy_keeps_a_cloud_first_turn_local(tmp_path: Path) -> None:
    dialog, local, remote = _dialog(tmp_path, policy=RoutingPolicy.CLOUD_FIRST)

    result = dialog.send("привет", privacy="local_only")

    assert result.ok
    assert result.answer == "from local"
    assert remote.requests == []


def test_unknown_privacy_mode_is_rejected(tmp_path: Path) -> None:
    dialog, _local, _remote = _dialog(tmp_path)

    with pytest.raises(ValueError, match="unknown privacy mode"):
        dialog.send("привет", privacy="classified")


def test_failover_completes_the_turn_on_the_next_provider(tmp_path: Path) -> None:
    dialog, local, remote = _dialog(
        tmp_path,
        policy=RoutingPolicy.FALLBACK,
        local_reply=ProviderUnavailableError("local model is down"),
        allow_fallback=True,
    )

    result = dialog.send("привет")

    assert result.ok
    assert result.answer == "from cloud"
    assert len(remote.requests) == 1


def test_a_turn_routed_to_a_remote_provider_still_cannot_call_unknown_tools(
    tmp_path: Path,
) -> None:
    """Routing never widens permissions: an unknown tool stays denied."""
    dialog, local, remote = _dialog(
        tmp_path,
        policy=RoutingPolicy.CHEAPEST,
        local_cost=3,
        remote_cost=1,
        remote_reply=structured_call("ghost_tool"),
    )

    result = dialog.send("вызови инструмент")

    assert result.status is not DialogStatus.SUCCESS
    # The call is recorded as refused: recorded, never executed.
    assert result.tool_calls[0].status != "success"
    assert len(remote.requests) == 1
