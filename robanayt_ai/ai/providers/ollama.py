"""Ollama (local) provider - stdlib HTTP only, no SDK, no model downloads.

Contract:

- Constructing the provider never opens a connection.
- `status(probe=False)` reports configuration only (used at startup).
- `status(probe=True)` performs one `GET /api/tags` - explicit user action.
- Connection, timeout and malformed-payload failures become typed errors;
  an unavailable Ollama never crashes the application.

Native tool calls (when `AIRequest.tools` is set):

- Declared tools are sent as Ollama `tools` (function schema).
- `message.tool_calls` is normalised into
  `AIResponse.metadata["tool_call"]`; only `name` and `arguments` survive.
- A malformed, ambiguous (more than one call) or unsupported call is dropped
  and logged - the ordinary answer is kept and nothing is executed.
- Streaming does not aggregate tool calls: `stream()` yields text only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping
from typing import Any

from robanayt_ai.ai.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    sanitize,
)
from robanayt_ai.ai.http import (
    KIND_AUTH,
    KIND_CONNECTION,
    KIND_HTTP,
    KIND_PARSE,
    KIND_RATE_LIMIT,
    KIND_TIMEOUT,
    HttpTransport,
    TransportError,
    UrllibTransport,
)
from robanayt_ai.ai.models import (
    AIChunk,
    AIRequest,
    AIResponse,
    ProviderStatus,
    Usage,
)
from robanayt_ai.ai.provider import LOCAL, ProviderCapabilities, ProviderProfile
from robanayt_ai.ai.tool_calls import (
    tool_call_from_message,
    tool_call_metadata,
    tool_payload,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaProvider:
    """Talks to a local Ollama server over its HTTP API."""

    name = "ollama"
    kind = LOCAL
    capabilities = ProviderCapabilities(streaming=True, local=True, tools=True)

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        *,
        timeout: float = 30.0,
        transport: HttpTransport | None = None,
        profile: ProviderProfile | None = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or ""
        self.timeout = timeout
        self._transport = transport or UrllibTransport()
        # Declared, not measured: local inference costs no per-request fee.
        self.profile = profile or ProviderProfile(
            cost=0, notes="local inference: no per-request fee"
        )

    def __repr__(self) -> str:
        return (
            f"OllamaProvider(base_url={self.base_url!r}, model={self.model!r}, "
            f"timeout={self.timeout})"
        )

    def status(self, probe: bool = False) -> ProviderStatus:
        """Configuration state; probes `/api/tags` only when asked."""
        if not self.model:
            return ProviderStatus(
                provider=self.name,
                configured=False,
                available=False,
                message="no model configured (ai.default_model / ai.ollama_model)",
            )

        if not probe:
            return ProviderStatus(
                provider=self.name,
                configured=True,
                available=True,
                message=f"configured for model {self.model!r}; availability not probed",
            )

        try:
            payload = self._transport.get_json(
                f"{self.base_url}/api/tags", timeout=self.timeout
            )
        except TransportError as exc:
            mapped = self._map_error(exc)
            return ProviderStatus(
                provider=self.name,
                configured=True,
                available=False,
                message=sanitize(f"probe failed: {mapped.raw_message}"),
                checked_at=time.time(),
            )

        models = self._extract_models(payload)
        if self.model in models:
            return ProviderStatus(
                provider=self.name,
                configured=True,
                available=True,
                message=f"model {self.model!r} present locally",
                checked_at=time.time(),
            )
        return ProviderStatus(
            provider=self.name,
            configured=True,
            available=False,
            message=f"model {self.model!r} not found on server "
            f"({len(models)} model(s) available)",
            checked_at=time.time(),
        )

    def generate(self, request: AIRequest) -> AIResponse:
        request.validate()
        model = self._resolve_model(request)
        payload = self._build_payload(request, model, stream=False)
        try:
            data = self._transport.post_json(
                f"{self.base_url}/api/chat", payload, timeout=self._timeout(request)
            )
        except TransportError as exc:
            raise self._map_error(exc, model=model) from exc
        return self._parse_response(data, model)

    def stream(self, request: AIRequest) -> Iterator[AIChunk]:
        request.validate()
        model = self._resolve_model(request)
        payload = self._build_payload(request, model, stream=True)
        try:
            lines = self._transport.post_stream_lines(
                f"{self.base_url}/api/chat", payload, timeout=self._timeout(request)
            )
            for line in lines:
                chunk = self._parse_chunk_line(line, model)
                if chunk is not None:
                    yield chunk
        except TransportError as exc:
            raise self._map_error(exc, model=model) from exc

    def _build_payload(
        self, request: AIRequest, model: str, *, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.as_dict() for message in request.messages],
            "stream": stream,
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options
        if request.tools:
            payload["tools"] = tool_payload(request.tools)
        return payload

    def _timeout(self, request: AIRequest) -> float:
        return request.timeout if request.timeout is not None else self.timeout

    def _resolve_model(self, request: AIRequest) -> str:
        model = request.model or self.model
        if not model:
            raise ProviderConfigurationError(
                "no model configured for the ollama provider", provider=self.name
            )
        return model

    def _parse_response(self, data: Any, model: str) -> AIResponse:
        if not isinstance(data, dict):
            raise ProviderResponseError(
                f"unexpected response type: {type(data).__name__}", provider=self.name
            )

        message = data.get("message")
        call: dict[str, Any] | None = None
        rejection: str | None = None
        if isinstance(message, dict):
            content = message.get("content")
            call, rejection = tool_call_from_message(message)
        else:
            content = data.get("response")
        if rejection:
            logger.warning("ollama tool call rejected: %s", rejection)

        if not isinstance(content, str):
            # A pure tool-call response may carry no text at all; a response
            # with neither text nor a usable call is a provider error.
            content = "" if call is not None else None
        if content is None:
            raise ProviderResponseError(
                "response has no text content", provider=self.name
            )

        usage = None
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
            total = None
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                total = prompt_tokens + completion_tokens
            usage = Usage(
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                completion_tokens=(
                    completion_tokens if isinstance(completion_tokens, int) else None
                ),
                total_tokens=total,
            )

        finish_reason = data.get("done_reason")
        if not isinstance(finish_reason, str) and data.get("done") is True:
            finish_reason = "stop"

        return AIResponse(
            content=content,
            model=data.get("model") or model,
            provider=self.name,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=usage,
            metadata=tool_call_metadata(call),
        )

    def _parse_chunk_line(self, line: str, model: str) -> AIChunk | None:
        try:
            import json

            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"malformed stream line: {exc}", provider=self.name
            ) from exc

        if not isinstance(data, dict):
            raise ProviderResponseError(
                "stream line is not an object", provider=self.name
            )
        if "error" in data:
            raise ProviderResponseError(
                sanitize(str(data.get("error"))), provider=self.name
            )

        message = data.get("message")
        delta = ""
        if isinstance(message, dict):
            value = message.get("content")
            delta = value if isinstance(value, str) else ""
        elif isinstance(data.get("response"), str):
            delta = data["response"]

        done = bool(data.get("done"))
        finish_reason = data.get("done_reason")
        return AIChunk(
            delta=delta,
            done=done,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            model=data.get("model") or model,
            provider=self.name,
        )

    def _extract_models(self, payload: Any) -> tuple[str, ...]:
        if not isinstance(payload, dict):
            return ()
        models = payload.get("models")
        if not isinstance(models, list):
            return ()
        names: list[str] = []
        for entry in models:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("model")
                if isinstance(name, str):
                    names.append(name)
        return tuple(names)

    def _map_error(self, exc: TransportError, *, model: str | None = None) -> ProviderError:
        detail = sanitize(exc.raw_message)
        if exc.kind == KIND_TIMEOUT:
            return ProviderTimeoutError(detail, provider=self.name, status=exc.status)
        if exc.kind == KIND_CONNECTION:
            return ProviderUnavailableError(detail, provider=self.name, status=exc.status)
        if exc.kind == KIND_AUTH:
            return ProviderAuthenticationError(
                detail, provider=self.name, status=exc.status
            )
        if exc.kind == KIND_RATE_LIMIT:
            return ProviderRateLimitError(detail, provider=self.name, status=exc.status)
        if exc.kind in {KIND_HTTP, KIND_PARSE}:
            return ProviderResponseError(detail, provider=self.name, status=exc.status)
        return ProviderError(detail, provider=self.name, status=exc.status)

    def describe(self) -> Mapping[str, Any]:
        """Non-sensitive description for diagnostics."""
        return {
            "provider": self.name,
            "profile": self.profile.as_dict(),
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model or None,
            "streaming": self.capabilities.streaming,
            "tools": self.capabilities.tools,
        }
