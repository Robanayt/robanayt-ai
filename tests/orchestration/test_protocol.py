"""Tool-call protocol: structured channel first, text channel only when enabled."""

from __future__ import annotations

import json

import pytest

from jarvis.ai.models import AIResponse, AIMessage
from jarvis.orchestration.protocol import (
    SOURCE_METADATA,
    SOURCE_TEXT,
    TOOL_CALL_KEY,
    ModelToolCall,
    text_protocol_example,
    tool_call_from_response,
)


def test_structured_call_is_extracted() -> None:
    response = AIResponse(
        content="anything",
        metadata={TOOL_CALL_KEY: {"name": "add_numbers", "arguments": {"a": 1, "b": 2}}},
    )

    call, error = tool_call_from_response(response)

    assert error is None
    assert call == ModelToolCall(name="add_numbers", arguments={"a": 1, "b": 2}, source=SOURCE_METADATA)


def test_structured_call_without_arguments() -> None:
    response = AIResponse(content="", metadata={TOOL_CALL_KEY: {"name": "current_time"}})

    call, error = tool_call_from_response(response)

    assert error is None
    assert call is not None and call.arguments == {}


def test_structured_call_rejects_permission_smuggling() -> None:
    response = AIResponse(
        content="",
        metadata={TOOL_CALL_KEY: {"name": "echo_text", "arguments": {}, "approved": True}},
    )

    call, error = tool_call_from_response(response)

    assert call is None
    assert "cannot grant permission" in (error or "")


def test_structured_call_rejects_unknown_fields() -> None:
    response = AIResponse(
        content="", metadata={TOOL_CALL_KEY: {"name": "ping", "extra": 1}}
    )

    call, error = tool_call_from_response(response)

    assert call is None
    assert "unsupported" in (error or "")


def test_structured_call_requires_a_mapping() -> None:
    response = AIResponse(content="", metadata={TOOL_CALL_KEY: "add_numbers"})

    call, error = tool_call_from_response(response)

    assert call is None and error is not None


def test_prose_is_never_a_tool_call() -> None:
    response = AIResponse(content="I could add 2 and 2 if you want.")

    call, error = tool_call_from_response(response, allow_text=True)

    assert call is None and error is None


def test_text_channel_is_ignored_by_default() -> None:
    payload = json.dumps({TOOL_CALL_KEY: {"name": "add_numbers", "arguments": {}}})
    response = AIResponse(content=f"```json\n{payload}\n```")

    call, error = tool_call_from_response(response)

    assert call is None and error is None


def test_text_channel_parses_the_explicit_block() -> None:
    payload = json.dumps({TOOL_CALL_KEY: {"name": "add_numbers", "arguments": {"a": 2}}})
    response = AIResponse(content=f"```json\n{payload}\n```")

    call, error = tool_call_from_response(response, allow_text=True)

    assert error is None
    assert call == ModelToolCall(name="add_numbers", arguments={"a": 2}, source=SOURCE_TEXT)


def test_text_channel_reports_malformed_json() -> None:
    response = AIResponse(content="```json\n{not json}\n```")

    call, error = tool_call_from_response(response, allow_text=True)

    assert call is None
    assert "malformed" in (error or "")


def test_text_channel_allows_non_call_json_blocks() -> None:
    response = AIResponse(content='```json\n{"answer": 4}\n```')

    call, error = tool_call_from_response(response, allow_text=True)

    assert call is None and error is None


def test_text_channel_rejects_forbidden_fields() -> None:
    payload = json.dumps(
        {TOOL_CALL_KEY: {"name": "echo_text", "arguments": {}, "force": True}}
    )
    response = AIResponse(content=f"```json\n{payload}\n```")

    call, error = tool_call_from_response(response, allow_text=True)

    assert call is None and "cannot grant permission" in (error or "")


def test_example_matches_the_parser() -> None:
    response = AIResponse(content=text_protocol_example())

    call, error = tool_call_from_response(response, allow_text=True)

    assert error is None
    assert call is not None and call.name == "add_numbers"


def test_messages_are_untouched_by_the_protocol() -> None:
    assert AIMessage.tool("result").role == "tool"
    with pytest.raises(Exception):
        AIMessage(role="root", content="x")
