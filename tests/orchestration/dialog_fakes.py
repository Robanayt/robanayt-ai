"""Helpers for dialog-orchestration tests: fake clock, scripted provider, wiring.

Imported through the package (`tests.orchestration.dialog_fakes`) and never
through a bare module name: several test packages in this project ship a
`conftest.py`, and importing one of those by name would pick the wrong file
depending on collection order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jarvis.ai.models import AIChunk, AIRequest, AIResponse, ProviderStatus
from jarvis.ai.provider import ProviderCapabilities
from jarvis.ai.router import ProviderRouter, RoutingPolicy
from jarvis.ai.service import AIService
from jarvis.core.config import MemoryConfig, OrchestrationConfig, ToolsConfig
from jarvis.memory import MemoryService
from jarvis.orchestration import DialogService
from jarvis.orchestration.protocol import TOOL_CALL_KEY
from jarvis.tools import build_tool_system
from jarvis.tools.models import ArgumentSpec, PermissionLevel, ToolSpec


class FakeClock:
    """Deterministic clock: moves only when a test says so."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedProvider:
    """Provider with scripted replies; records every request it receives."""

    def __init__(
        self,
        replies: list[AIResponse | Exception],
        *,
        name: str = "scripted",
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.kind = "local"
        self.capabilities = ProviderCapabilities(streaming=True, local=True)
        self.replies = list(replies)
        self.enabled = enabled
        self.requests: list[AIRequest] = []

    def status(self, probe: bool = False) -> ProviderStatus:
        return ProviderStatus(
            provider=self.name,
            configured=self.enabled,
            available=self.enabled,
            message="scripted provider",
        )

    def generate(self, request: AIRequest) -> AIResponse:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("scripted provider ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def stream(self, request: AIRequest) -> Any:
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else AIResponse(content="")
        if isinstance(reply, Exception):
            raise reply
        yield AIChunk(delta=reply.content, model=request.model, provider=self.name)
        yield AIChunk("", done=True, finish_reason="stop", provider=self.name)


def answer(text: str, *, model: str = "test-model") -> AIResponse:
    return AIResponse(content=text, model=model, provider="scripted")


def structured_call(name: str, arguments: dict[str, Any] | None = None) -> AIResponse:
    """A tool call through the structured channel (response metadata)."""
    return AIResponse(
        content="",
        model="test-model",
        provider="scripted",
        metadata={TOOL_CALL_KEY: {"name": name, "arguments": arguments or {}}},
    )


def text_call(name: str, arguments: dict[str, Any] | None = None) -> AIResponse:
    """A tool call through the explicit fenced-JSON channel."""
    payload = json.dumps(
        {TOOL_CALL_KEY: {"name": name, "arguments": arguments or {}}}, ensure_ascii=False
    )
    return AIResponse(content=f"```json\n{payload}\n```", model="test-model", provider="scripted")


def ai_service(provider: ScriptedProvider, *, enabled: bool = True) -> AIService:
    router = ProviderRouter(
        [provider], default_provider=provider.name, policy=RoutingPolicy.LOCAL_FIRST
    )
    return AIService(router, enabled=enabled, default_model="test-model")


def memory_service(tmp_path: Path, *, enabled: bool = True) -> MemoryService:
    return MemoryService(
        MemoryConfig(directory=str(tmp_path / "memory"), enabled=enabled)
    )


def make_dialog(
    tmp_path: Path,
    replies: list[AIResponse | Exception],
    *,
    config: OrchestrationConfig | None = None,
    clock: FakeClock | None = None,
    memory_enabled: bool = True,
    ai_enabled: bool = True,
):
    """Wire a DialogService with fakes. No network, no real provider."""
    clock = clock or FakeClock()
    provider = ScriptedProvider(replies)
    memory = memory_service(tmp_path, enabled=memory_enabled)
    tools = build_tool_system(ToolsConfig(), clock=clock)
    dialog = DialogService(
        ai_service(provider, enabled=ai_enabled),
        memory,
        tools,
        config or OrchestrationConfig(),
        clock=clock,
    )
    return dialog, provider, memory, tools, clock


def register_tool(
    tools: Any,
    name: str,
    handler: Any,
    *,
    permission: PermissionLevel = PermissionLevel.SAFE,
    arguments: tuple["ArgumentSpec", ...] = (),
) -> None:
    """Register a test tool explicitly (registration never runs it)."""
    spec = ToolSpec(
        name=name,
        description=f"test tool {name}",
        permission=permission,
        side_effects=permission is not PermissionLevel.SAFE,
        arguments=arguments,
    )
    tools.registry.register(spec, handler)


__all__ = [
    "FakeClock",
    "PermissionLevel",
    "ScriptedProvider",
    "ai_service",
    "answer",
    "make_dialog",
    "memory_service",
    "register_tool",
    "structured_call",
    "text_call",
]
