"""Streaming abstraction."""

from __future__ import annotations

from jarvis.ai.models import AIChunk, AIResponse
from jarvis.ai.streaming import (
    StreamAccumulator,
    accumulate,
    as_stream,
    collect_text,
    iter_text,
)


def _chunks() -> list[AIChunk]:
    return [
        AIChunk(delta="Hello", model="m", provider="p"),
        AIChunk(delta=" "),
        AIChunk(delta="world"),
        AIChunk("", done=True, finish_reason="stop"),
    ]


def test_iter_text_skips_empty_deltas() -> None:
    assert list(iter_text(_chunks())) == ["Hello", " ", "world"]


def test_collect_text_joins_deltas() -> None:
    assert collect_text(_chunks()) == "Hello world"


def test_as_stream_wraps_complete_response() -> None:
    response = AIResponse(content="done", model="m", provider="p")

    chunks = list(as_stream(response))

    assert len(chunks) == 1
    assert chunks[0].delta == "done"
    assert chunks[0].done is True
    assert chunks[0].finish_reason == "stop"


def test_accumulator_tracks_text_and_metadata() -> None:
    accumulator = StreamAccumulator()

    for chunk in _chunks():
        accumulator.push(chunk)

    assert accumulator.text == "Hello world"
    assert accumulator.finish_reason == "stop"
    assert accumulator.done is True
    assert accumulator.model == "m"
    assert accumulator.provider == "p"


def test_accumulator_to_response() -> None:
    accumulator = StreamAccumulator(model="m", provider="p")
    accumulator.push(AIChunk(delta="a"))
    accumulator.push(AIChunk(delta="b", done=True, finish_reason="length"))

    response = accumulator.to_response()

    assert response.content == "ab"
    assert response.finish_reason == "length"
    assert response.provider == "p"


def test_accumulate_helper() -> None:
    response = accumulate(_chunks(), model="m", provider="p")

    assert response.content == "Hello world"
    assert response.finish_reason == "stop"


def test_empty_stream_produces_empty_response() -> None:
    response = accumulate([])

    assert response.content == ""
    assert response.finish_reason is None
