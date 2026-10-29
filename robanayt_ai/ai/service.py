"""AIService - the single entry point to the intelligence layer.

Responsibilities: validate requests, ask the router for a provider, execute
with conservative timeouts/retries, normalise the result, and expose status.
Callers never see provider-specific shapes or errors they cannot handle.

Retry policy (deliberately conservative):

- Retried: timeout, unavailable, rate limit.
- Never retried: authentication, configuration, malformed response,
  invalid request, disabled AI.
- Attempt count is bounded by `max_retries` (default 0 = a single attempt).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

from robanayt_ai.ai.context import ConversationContext
from robanayt_ai.ai.errors import (
    AIDisabledError,
    AIRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    sanitize,
)
from robanayt_ai.ai.models import (
    AIChunk,
    AIRequest,
    AIResponse,
    AIMessage,
    AITool,
    ProviderStatus,
)
from robanayt_ai.ai.router import ProviderRouter, RoutingDecision
from robanayt_ai.ai.streaming import StreamAccumulator, as_stream

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = (
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderRateLimitError,
)


class AIService:
    """High-level, provider-agnostic AI facade."""

    def __init__(
        self,
        router: ProviderRouter,
        *,
        enabled: bool = True,
        default_model: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 0,
        retry_delay_seconds: float = 0.0,
        allow_fallback: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.router = router
        self.enabled = enabled
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        # Bounded failover: each alternative provider is tried at most once,
        # only after the chosen one failed with a retryable error.
        self.allow_fallback = bool(allow_fallback)
        self._sleeper = sleeper

    def describe(self) -> dict[str, Any]:
        """Diagnostics summary: providers, policy, no credentials."""
        summary = self.router.describe()
        summary.update(
            {
                "enabled": self.enabled,
                "default_model": self.default_model,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "allow_fallback": self.allow_fallback,
                "routing": self.explain(),
            }
        )
        return summary

    def explain(self, request: AIRequest | None = None) -> dict[str, Any]:
        """What would be used for a request, and why not the others."""
        summary = self.router.explain(request)
        summary["enabled"] = self.enabled
        summary["default_model"] = self.default_model
        summary["allow_fallback"] = self.allow_fallback
        if not self.enabled:
            summary["reason"] = "AI layer is disabled (ai.enabled=false)"
        return summary

    def status(self, probe: bool = False) -> tuple[ProviderStatus, ...]:
        """Provider statuses. `probe=True` may contact a provider."""
        return self.router.status(probe=probe)

    def build_request(
        self,
        *,
        prompt: str | None = None,
        messages: Sequence[AIMessage] | None = None,
        system: str | None = None,
        context: ConversationContext | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        tools: Sequence[AITool] | None = None,
    ) -> AIRequest:
        """Assemble a validated request from either a prompt or messages."""
        if context is not None and messages:
            raise AIRequestError("pass either context or messages, not both")

        if messages is not None:
            prepared: tuple[AIMessage, ...] = tuple(messages)
        else:
            prepared = (AIMessage.user(prompt),) if prompt is not None else ()

        if context is not None:
            prepared = context.to_messages() + prepared

        if system:
            prepared = (AIMessage.system(system),) + prepared

        request = AIRequest(
            messages=prepared,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            metadata=dict(metadata or {}),
            tools=tuple(tools or ()),
        )
        request.validate()
        return request

    def generate(
        self,
        request: AIRequest | None = None,
        *,
        provider: str | None = None,
        privacy: str | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Run one generation and return a normalised response."""
        prepared = self._prepare(request, **kwargs)
        decision = self.router.select(prepared, provider=provider, privacy=privacy)
        return self._execute(decision, prepared)

    def stream(
        self,
        request: AIRequest | None = None,
        *,
        provider: str | None = None,
        privacy: str | None = None,
        **kwargs: Any,
    ) -> Iterator[AIChunk]:
        """Stream a response; falls back to one chunk if unsupported.

        A stream is never re-routed mid-flight: partial output cannot be
        undone, so failover applies to `generate()` only.
        """
        prepared = self._prepare(request, **kwargs)
        decision = self.router.select(prepared, provider=provider, privacy=privacy)
        target = decision.provider

        if not getattr(target.capabilities, "streaming", False):
            response = self._execute(decision, prepared)
            yield from as_stream(response)
            return

        yield from self._execute_stream(decision, prepared)

    def _prepare(self, request: AIRequest | None, **kwargs: Any) -> AIRequest:
        if not self.enabled:
            raise AIDisabledError("AI layer is disabled (ai.enabled=false)")
        prepared = request if request is not None else self.build_request(**kwargs)
        prepared.validate()
        return prepared.with_overrides(model=self.default_model, timeout=self.timeout)

    def _execute(self, decision: RoutingDecision, request: AIRequest) -> AIResponse:
        provider = decision.provider
        try:
            return self._run(provider, request)
        except RETRYABLE_ERRORS as exc:
            if not self.allow_fallback or not decision.chain:
                raise
            last_error: ProviderError = exc

        # Bounded failover: one attempt per alternative, never a loop, and only
        # for retryable failures (auth/config/response errors are final).
        for name in decision.chain:
            alternative = self.router.get(name)
            try:
                response = self._run(alternative, request)
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                continue
            logger.warning(
                "provider %s failed; routed to %s instead: %s",
                provider.name,
                name,
                sanitize(str(last_error)),
            )
            return self._with_failover_note(
                self._normalise(response, alternative.name),
                from_provider=provider.name,
                to_provider=name,
                reason=str(last_error),
            )

        logger.error(
            "provider %s failed and every alternative failed too: %s",
            provider.name,
            sanitize(str(last_error)),
        )
        raise last_error

    def _run(self, provider: Any, request: AIRequest) -> AIResponse:
        """One provider with the conservative retry policy. No re-routing."""
        attempts = self.max_retries + 1
        last_error: ProviderError | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = provider.generate(request)
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                logger.warning(
                    "provider %s failed (attempt %s/%s): %s",
                    provider.name,
                    attempt,
                    attempts,
                    sanitize(str(exc)),
                )
                self._delay()
                continue
            except ProviderError as exc:
                logger.error("provider %s error: %s", provider.name, sanitize(str(exc)))
                raise

            return self._normalise(response, provider.name)

        assert last_error is not None
        logger.error(
            "provider %s failed after %s attempt(s): %s",
            provider.name,
            attempts,
            sanitize(str(last_error)),
        )
        raise last_error

    @staticmethod
    def _with_failover_note(
        response: AIResponse,
        *,
        from_provider: str,
        to_provider: str,
        reason: str,
    ) -> AIResponse:
        """Record that a different provider answered - never hidden."""
        metadata = dict(response.metadata or {})
        metadata["routing_failover"] = {
            "from": from_provider,
            "to": to_provider,
            "reason": sanitize(reason),
        }
        return replace(response, metadata=metadata)

    def _execute_stream(
        self, decision: RoutingDecision, request: AIRequest
    ) -> Iterator[AIChunk]:
        provider = decision.provider
        accumulator = StreamAccumulator(provider=provider.name)
        try:
            for chunk in provider.stream(request):
                accumulator.push(chunk)
                yield chunk
        except ProviderError as exc:
            logger.error(
                "provider %s stream error: %s", provider.name, sanitize(str(exc))
            )
            raise
        logger.debug(
            "stream finished: %s chars, finish_reason=%s",
            len(accumulator.text),
            accumulator.finish_reason,
        )

    def _normalise(self, response: AIResponse, provider_name: str) -> AIResponse:
        if response.provider and response.provider != provider_name:
            return response
        return replace(response, provider=provider_name)

    def _delay(self) -> None:
        if self.retry_delay_seconds > 0:
            self._sleeper(self.retry_delay_seconds)
