"""Shared fakes for AI-layer tests. No network, no credentials."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

from robanayt_ai.ai.http import KIND_CONNECTION, TransportError
from robanayt_ai.ai.models import AIChunk, AIRequest, AIResponse, ProviderStatus
from robanayt_ai.ai.provider import ProviderCapabilities, ProviderProfile


class FakeTransport:
    """Scripted transport: canned results or a scripted failure."""

    def __init__(
        self,
        *,
        get_result: Any = None,
        post_result: Any = None,
        stream_lines: list[str] | None = None,
        error: TransportError | None = None,
    ) -> None:
        self.get_result = get_result
        self.post_result = post_result
        self.stream_lines = stream_lines or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers or {})})
        if self.error:
            raise self.error
        return self.get_result

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        self.calls.append(
            {
                "method": "POST",
                "url": url,
                "payload": dict(payload),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        if self.error:
            raise self.error
        return self.post_result

    def post_stream_lines(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Iterator[str]:
        self.calls.append(
            {
                "method": "STREAM",
                "url": url,
                "payload": dict(payload),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        if self.error:
            raise self.error
        for line in self.stream_lines:
            yield line


def offline_transport() -> FakeTransport:
    """Transport whose failure means 'somebody tried to use the network'."""
    return FakeTransport(
        error=TransportError(KIND_CONNECTION, "network access attempted in test")
    )


class FakeProvider:
    """Deterministic provider used by router/service tests."""

    def __init__(
        self,
        name: str = "fake",
        *,
        kind: str = "local",
        configured: bool = True,
        available: bool = True,
        message: str = "fake provider",
        reply: str = "fake reply",
        streaming: bool = True,
        failures: list[Exception] | None = None,
        profile: ProviderProfile | None = None,
        model: str = "",
    ) -> None:
        self.name = name
        self.kind = kind
        self.capabilities = ProviderCapabilities(streaming=streaming, local=kind == "local")
        # `None` means "declares nothing": the router then derives a default.
        self.profile = profile
        self.model = model
        self._configured = configured
        self._available = available
        self._message = message
        self.reply = reply
        self.failures = list(failures or [])
        self.calls: list[AIRequest] = []

    def status(self, probe: bool = False) -> ProviderStatus:
        return ProviderStatus(
            provider=self.name,
            configured=self._configured,
            available=self._available,
            message=self._message,
        )

    def generate(self, request: AIRequest) -> AIResponse:
        self.calls.append(request)
        if self.failures:
            raise self.failures.pop(0)
        return AIResponse(content=self.reply, model=request.model, provider=self.name)

    def stream(self, request: AIRequest) -> Iterator[AIChunk]:
        self.calls.append(request)
        if self.failures:
            raise self.failures.pop(0)
        for word in self.reply.split():
            yield AIChunk(delta=word + " ", model=request.model, provider=self.name)
        yield AIChunk("", done=True, finish_reason="stop", provider=self.name)


def ollama_chat_response(content: str, model: str = "test-model") -> dict[str, Any]:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 7,
    }


def openai_chat_response(content: str, model: str = "gpt-test") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


def sse_lines(content_parts: list[str], model: str = "gpt-test") -> list[str]:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": part}, "finish_reason": None}], "model": model})
        for part in content_parts
    ]
    lines.append(
        "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "model": model})
    )
    lines.append("data: [DONE]")
    return lines
