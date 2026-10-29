"""Streaming helpers.

Consumers iterate `AIChunk` objects and never learn which provider produced
them. Providers without streaming support can still be consumed through
`as_stream()`, which wraps a complete response in a single final chunk.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from jarvis.ai.models import AIChunk, AIResponse


def iter_text(chunks: Iterable[AIChunk]) -> Iterator[str]:
    """Yield non-empty text deltas only."""
    for chunk in chunks:
        if chunk.delta:
            yield chunk.delta


def collect_text(chunks: Iterable[AIChunk]) -> str:
    """Concatenate every delta into the full text."""
    return "".join(iter_text(chunks))


def as_stream(
    response: AIResponse,
    *,
    finish_reason: str | None = None,
) -> Iterator[AIChunk]:
    """Expose a non-streaming response as a one-chunk stream."""
    yield AIChunk(
        delta=response.content,
        done=True,
        finish_reason=finish_reason or response.finish_reason or "stop",
        model=response.model,
        provider=response.provider,
    )


class StreamAccumulator:
    """Collects chunks into text plus terminal metadata."""

    def __init__(self, *, model: str | None = None, provider: str | None = None) -> None:
        self.model = model
        self.provider = provider
        self._parts: list[str] = []
        self._finish_reason: str | None = None
        self._done = False

    @property
    def text(self) -> str:
        return "".join(self._parts)

    @property
    def finish_reason(self) -> str | None:
        return self._finish_reason

    @property
    def done(self) -> bool:
        return self._done

    def push(self, chunk: AIChunk) -> str:
        """Add a chunk; returns the accumulated text so far."""
        if chunk.delta:
            self._parts.append(chunk.delta)
        if chunk.model and not self.model:
            self.model = chunk.model
        if chunk.provider and not self.provider:
            self.provider = chunk.provider
        if chunk.finish_reason:
            self._finish_reason = chunk.finish_reason
        if chunk.done:
            self._done = True
        return self.text

    def to_response(self) -> AIResponse:
        """Convert accumulated chunks into a normal response object."""
        return AIResponse(
            content=self.text,
            model=self.model,
            provider=self.provider,
            finish_reason=self._finish_reason,
        )


def accumulate(
    chunks: Iterable[AIChunk],
    *,
    model: str | None = None,
    provider: str | None = None,
) -> AIResponse:
    """Fold a chunk iterable into a single `AIResponse`."""
    accumulator = StreamAccumulator(model=model, provider=provider)
    for chunk in chunks:
        accumulator.push(chunk)
    return accumulator.to_response()
