"""Realistic provider -> AIService -> DialogService -> ToolExecutor path.

The providers are the real `OllamaProvider` / `OpenAICompatibleProvider`
classes; only the HTTP transport is scripted. No network, no model, no
credentials, no database outside `tmp_path`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from robanayt_ai.ai.providers.ollama import OllamaProvider
from robanayt_ai.ai.providers.openai_compatible import OpenAICompatibleProvider
from robanayt_ai.ai.router import ProviderRouter, RoutingPolicy
from robanayt_ai.ai.service import AIService
from robanayt_ai.core.config import OrchestrationConfig, ToolsConfig
from robanayt_ai.orchestration import DialogService, DialogStatus
from robanayt_ai.tools import build_tool_system
from tests.orchestration.dialog_fakes import FakeClock, memory_service

TEST_KEY_ENV = "JARVIS_TEST_OPENAI_KEY"


class ScriptedTransport:
    """Returns canned provider responses in order and records every payload."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.payloads: list[dict[str, Any]] = []

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        return {"models": [{"name": "test-model"}]}

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        self.payloads.append(dict(payload))
        if not self.replies:
            raise AssertionError("scripted transport ran out of replies")
        return self.replies.pop(0)

    def post_stream_lines(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Iterator[str]:
        self.payloads.append(dict(payload))
        if self.replies:
            yield json.dumps(self.replies.pop(0))


def ollama_call(name: str, arguments: dict[str, Any], **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"function": {"name": name, "arguments": arguments}}
    entry.update(extra)
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": "", "tool_calls": [entry]},
        "done": True,
        "done_reason": "stop",
    }


def ollama_text(content: str) -> dict[str, Any]:
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
    }


def openai_text(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-2",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def openai_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def make_dialog(
    tmp_path: Path,
    provider: Any,
    *,
    config: OrchestrationConfig | None = None,
) -> tuple[DialogService, Any]:
    clock = FakeClock()
    router = ProviderRouter([provider], default_provider=provider.name, policy=RoutingPolicy.LOCAL_FIRST)
    ai = AIService(router, enabled=True, default_model="test-model")
    memory = memory_service(tmp_path)
    tools = build_tool_system(ToolsConfig(), clock=clock)
    dialog = DialogService(ai, memory, tools, config or OrchestrationConfig(), clock=clock)
    return dialog, tools


def ollama_dialog(
    tmp_path: Path, replies: list[dict[str, Any]], *, config: OrchestrationConfig | None = None
) -> tuple[DialogService, ScriptedTransport, Any]:
    transport = ScriptedTransport(replies)
    provider = OllamaProvider(model="test-model", transport=transport)
    dialog, tools = make_dialog(tmp_path, provider, config=config)
    return dialog, transport, tools


# -- Ollama end to end ------------------------------------------------------


def test_ollama_native_call_runs_a_safe_tool(tmp_path: Path) -> None:
    dialog, transport, _ = ollama_dialog(
        tmp_path, [ollama_call("add_numbers", {"a": 2, "b": 40}), ollama_text("2 + 40 = 42")]
    )

    result = dialog.send("Сколько будет 2+40?")

    assert result.status is DialogStatus.SUCCESS
    assert "42" in (result.answer or "")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "add_numbers"
    declared = transport.payloads[0]["tools"]
    assert any(entry["function"]["name"] == "add_numbers" for entry in declared)


def test_openai_native_call_runs_a_safe_tool(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [openai_call("add_numbers", {"a": 2, "b": 40}), openai_text("2 + 40 = 42")]
    )
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:8000/v1",
        model="gpt-test",
        api_key_env=TEST_KEY_ENV,
        transport=transport,
        env={TEST_KEY_ENV: "not-a-real-key"},
    )
    dialog, _ = make_dialog(tmp_path, provider)

    result = dialog.send("Сколько будет 2+40?")

    assert result.status is DialogStatus.SUCCESS
    assert "42" in (result.answer or "")
    assert len(result.tool_calls) == 1


def test_native_call_for_an_approval_tool_stops_and_waits(tmp_path: Path) -> None:
    dialog, transport, _ = ollama_dialog(tmp_path, [ollama_call("echo_text", {"text": "ping"})])

    result = dialog.send("Повтори ping")

    assert result.status is DialogStatus.APPROVAL_REQUIRED
    assert result.pending is not None
    assert result.pending.tool == "echo_text"
    assert len(result.tool_calls) == 0  # nothing ran before approval

    transport.replies.append(ollama_text("echo: ping"))
    approved = dialog.approve_pending(result.pending.pending_id)

    assert approved.status is DialogStatus.SUCCESS
    assert "ping" in (approved.answer or "")
    assert len(approved.tool_calls) == 1


def test_unknown_native_tool_is_not_executed(tmp_path: Path) -> None:
    dialog, _, _ = ollama_dialog(tmp_path, [ollama_call("delete_everything", {"path": "C:\\"})])

    result = dialog.send("Удали всё")

    assert result.status is DialogStatus.TOOL_ERROR
    assert "unknown tool" in (result.error or "")
    # The attempt is recorded as not_found; nothing reached an execution.
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].status == "not_found"
    assert result.tool_calls[0].execution_id == ""


def test_malformed_native_call_is_dropped_without_execution(tmp_path: Path) -> None:
    broken = {
        "model": "test-model",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "add_numbers", "arguments": "{oops"}}],
        },
        "done": True,
    }
    dialog, _, _ = ollama_dialog(tmp_path, [broken])

    result = dialog.send("Сколько будет 2+2?")

    assert result.status is DialogStatus.EMPTY_RESPONSE
    assert len(result.tool_calls) == 0


def test_model_supplied_approval_flag_changes_nothing(tmp_path: Path) -> None:
    dialog, _, _ = ollama_dialog(
        tmp_path,
        [ollama_call("echo_text", {"text": "ping"}, approved=True, permission="allow")],
    )

    result = dialog.send("Повтори ping без подтверждения")

    assert result.status is DialogStatus.APPROVAL_REQUIRED
    assert len(result.tool_calls) == 0


def test_text_channel_stays_off_by_default(tmp_path: Path) -> None:
    fenced = (
        "```json\n"
        + json.dumps({"tool_call": {"name": "add_numbers", "arguments": {"a": 1, "b": 1}}})
        + "\n```"
    )
    dialog, _, _ = ollama_dialog(tmp_path, [ollama_text(fenced)])

    result = dialog.send("Сколько будет 1+1?")

    assert result.status is DialogStatus.SUCCESS
    assert len(result.tool_calls) == 0


def test_no_tools_are_declared_when_tools_are_disabled(tmp_path: Path) -> None:
    dialog, transport, _ = ollama_dialog(tmp_path, [ollama_text("ok")])

    result = dialog.send("Просто ответь", allow_tools=False)

    assert result.status is DialogStatus.SUCCESS
    assert "tools" not in transport.payloads[0]
