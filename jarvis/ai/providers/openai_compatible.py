"""Generic OpenAI-compatible HTTP provider.

No official SDK, no vendor assumption: any endpoint exposing an
OpenAI-style `/chat/completions` contract works.

Security:

- The API key is never stored on the provider, only the *name* of the
  environment variable that holds it; the value is resolved per request.
- Keys, `Authorization` headers and organizations never reach logs, errors
  or status dictionaries.
- The key is not required to construct the provider or to start JARVIS; a
  missing key simply makes the provider "not available".

Native tool calls (when `AIRequest.tools` is set):

- Declared tools are sent as `tools` (function schema).
- `choices[0].message.tool_calls` is normalised into
  `AIResponse.metadata["tool_call"]`; only `name` and `arguments` survive,
  so nothing provider- or permission-specific reaches the orchestrator.
- A malformed, ambiguous (more than one call) or unsupported call is dropped
  and logged - the ordinary answer is kept and nothing is executed.
- Streaming does not aggregate tool calls: `stream()` yields text only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlparse

from jarvis.ai.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    sanitize,
)
from jarvis.ai.http import (
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
from jarvis.ai.models import AIChunk, AIRequest, AIResponse, ProviderStatus, Usage
from jarvis.ai.provider import REMOTE, ProviderCapabilities, ProviderProfile
from jarvis.ai.tool_calls import (
    tool_call_from_message,
    tool_call_metadata,
    tool_payload,
)

logger = logging.getLogger(__name__)

DEFAULT_API_KEY_ENV = "JARVIS_OPENAI_API_KEY"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


class OpenAICompatibleProvider:
    """Provider for OpenAI-compatible chat/completions endpoints."""

    name = "openai_compatible"
    kind = REMOTE
    capabilities = ProviderCapabilities(streaming=True, local=False, tools=True)

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        organization: str = "",
        timeout: float = 30.0,
        allow_remote: bool = False,
        transport: HttpTransport | None = None,
        env: Mapping[str, str] | None = None,
        profile: ProviderProfile | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key_env = api_key_env or DEFAULT_API_KEY_ENV
        self.organization = organization
        self.timeout = timeout
        self.allow_remote = allow_remote
        self._transport = transport or UrllibTransport()
        self._env = env
        # Declared, not measured: a hosted API is billed per request.
        self.profile = profile or ProviderProfile(
            cost=2, notes="remote API: billed per request"
        )

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleProvider(base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key_env={self.api_key_env!r}, "
            f"allow_remote={self.allow_remote})"
        )

    @property
    def _environment(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    def _api_key(self) -> str:
        return (self._environment.get(self.api_key_env) or "").strip()

    def _is_local(self) -> bool:
        try:
            host = urlparse(self.base_url).hostname or ""
        except ValueError:
            return False
        return host.lower() in LOCAL_HOSTS

    def status(self, probe: bool = False) -> ProviderStatus:
        """Configuration state; a probe is only performed when asked."""
        if not self.base_url:
            return ProviderStatus(
                provider=self.name,
                configured=False,
                available=False,
                message="base_url is not configured (ai.openai_base_url)",
            )
        if not self.model:
            return ProviderStatus(
                provider=self.name,
                configured=False,
                available=False,
                message="no model configured (ai.openai_model)",
            )
        if not self._api_key():
            return ProviderStatus(
                provider=self.name,
                configured=True,
                available=False,
                message=f"API key missing: set {self.api_key_env}",
            )
        if not self.allow_remote and not self._is_local():
            return ProviderStatus(
                provider=self.name,
                configured=True,
                available=False,
                message="remote calls disabled (ai.allow_remote=false)",
            )

        if not probe:
            return ProviderStatus(
                provider=self.name,
                configured=True,
                available=True,
                message="configured; availability not probed",
            )

        try:
            self._transport.get_json(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=self.timeout,
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
        return ProviderStatus(
            provider=self.name,
            configured=True,
            available=True,
            message="endpoint reachable",
            checked_at=time.time(),
        )

    def generate(self, request: AIRequest) -> AIResponse:
        request.validate()
        model = self._resolve_model(request)
        payload = self._build_payload(request, model, stream=False)
        try:
            data = self._transport.post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers=self._headers(),
                timeout=self._timeout(request),
            )
        except TransportError as exc:
            raise self._map_error(exc) from exc
        return self._parse_response(data, model)

    def stream(self, request: AIRequest) -> Iterator[AIChunk]:
        request.validate()
        model = self._resolve_model(request)
        payload = self._build_payload(request, model, stream=True)
        try:
            lines = self._transport.post_stream_lines(
                f"{self.base_url}/chat/completions",
                payload,
                headers=self._headers(),
                timeout=self._timeout(request),
            )
            for line in lines:
                chunk = self._parse_chunk_line(line, model)
                if chunk is not None:
                    yield chunk
        except TransportError as exc:
            raise self._map_error(exc) from exc

    def _headers(self) -> dict[str, str]:
        key = self._api_key()
        if not key:
            raise ProviderConfigurationError(
                f"API key missing: set {self.api_key_env}", provider=self.name
            )
        headers = {"Authorization": f"Bearer {key}"}
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _build_payload(
        self, request: AIRequest, model: str, *, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.as_dict() for message in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = tool_payload(request.tools)
        return payload

    def _timeout(self, request: AIRequest) -> float:
        return request.timeout if request.timeout is not None else self.timeout

    def _resolve_model(self, request: AIRequest) -> str:
        model = request.model or self.model
        if not model:
            raise ProviderConfigurationError(
                "no model configured for the openai-compatible provider",
                provider=self.name,
            )
        return model

    def _parse_response(self, data: Any, model: str) -> AIResponse:
        if not isinstance(data, dict):
            raise ProviderResponseError(
                f"unexpected response type: {type(data).__name__}", provider=self.name
            )
        if "error" in data:
            raise ProviderResponseError(
                sanitize(self._error_text(data["error"])), provider=self.name
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError("response has no choices", provider=self.name)

        first = choices[0]
        if not isinstance(first, dict):
            raise ProviderResponseError("malformed choice entry", provider=self.name)

        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        call, rejection = (
            tool_call_from_message(message) if isinstance(message, dict) else (None, None)
        )
        if rejection:
            logger.warning("openai-compatible tool call rejected: %s", rejection)
        if content is None:
            content = first.get("text")
        if not isinstance(content, str):
            # A pure tool-call response may carry no text at all; a response
            # with neither text nor a usable call is a provider error.
            content = "" if call is not None else None
        if content is None:
            raise ProviderResponseError(
                "choice has no text content", provider=self.name
            )

        usage = None
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            usage = Usage(
                prompt_tokens=_as_int(raw_usage.get("prompt_tokens")),
                completion_tokens=_as_int(raw_usage.get("completion_tokens")),
                total_tokens=_as_int(raw_usage.get("total_tokens")),
            )

        return AIResponse(
            content=content,
            model=data.get("model") or model,
            provider=self.name,
            finish_reason=_as_str(first.get("finish_reason")),
            usage=usage,
            metadata=tool_call_metadata(call),
        )

    def _parse_chunk_line(self, line: str, model: str) -> AIChunk | None:
        payload = line.strip()
        if payload.startswith("data:"):
            payload = payload[5:].strip()
        if not payload or payload == "[DONE]":
            return AIChunk(delta="", done=True, finish_reason="stop", model=model, provider=self.name)
        try:
            data = json.loads(payload)
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
                sanitize(self._error_text(data["error"])), provider=self.name
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        first = choices[0]
        if not isinstance(first, dict):
            return None
        delta = first.get("delta")
        text = delta.get("content") if isinstance(delta, dict) else None
        if text is None:
            text = first.get("text")
        return AIChunk(
            delta=text if isinstance(text, str) else "",
            done=_as_str(first.get("finish_reason")) is not None,
            finish_reason=_as_str(first.get("finish_reason")),
            model=_as_str(data.get("model")) or model,
            provider=self.name,
        )

    def _error_text(self, error: Any) -> str:
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or "provider error"
            return str(message)
        return str(error)

    def _map_error(self, exc: TransportError) -> ProviderError:
        detail = sanitize(exc.raw_message)
        if exc.kind == KIND_TIMEOUT:
            return ProviderTimeoutError(detail, provider=self.name, status=exc.status)
        if exc.kind == KIND_CONNECTION:
            return ProviderUnavailableError(
                detail, provider=self.name, status=exc.status
            )
        if exc.kind == KIND_AUTH:
            return ProviderAuthenticationError(
                "authentication failed (credentials rejected)",
                provider=self.name,
                status=exc.status,
            )
        if exc.kind == KIND_RATE_LIMIT:
            return ProviderRateLimitError(
                "rate limited by provider", provider=self.name, status=exc.status
            )
        if exc.kind in {KIND_HTTP, KIND_PARSE}:
            return ProviderResponseError(detail, provider=self.name, status=exc.status)
        return ProviderError(detail, provider=self.name, status=exc.status)

    def describe(self) -> Mapping[str, Any]:
        """Non-sensitive description for diagnostics (never the key)."""
        return {
            "provider": self.name,
            "profile": self.profile.as_dict(),
            "kind": self.kind,
            "base_url": self.base_url or None,
            "model": self.model or None,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(self._api_key()),
            "allow_remote": self.allow_remote,
            "streaming": self.capabilities.streaming,
            "tools": self.capabilities.tools,
        }


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None
