"""Native provider tool calls -> the shared `metadata["tool_call"]` contract.

No network, no model, no credentials: every provider runs against a scripted
transport. These tests cover normalisation only; the orchestrator side of the
same path lives in `tests/orchestration/test_provider_tool_calls.py`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from tests.ai.fakes import (
    FakeTransport,
    ollama_chat_response,
    openai_chat_response,
)

from jarvis.ai.errors import AIRequestError, ProviderResponseError
from jarvis.ai.models import AIRequest, AIMessage, AITool
from jarvis.ai.providers.ollama import OllamaProvider
from jarvis.ai.providers.openai_compatible import OpenAICompatibleProvider
from jarvis.ai.tool_calls import TOOL_CALL_KEY, tool_call_from_message
from jarvis.orchestration.protocol import tool_declarations

TEST_KEY_ENV = "JARVIS_TEST_OPENAI_KEY"


# -- helpers ---------------------------------------------------------------


def ollama_provider(transport: FakeTransport) -> OllamaProvider:
    return OllamaProvider(model="test-model", transport=transport)


def openai_provider(transport: FakeTransport) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://127.0.0.1:8000/v1",
        model="gpt-test",
        api_key_env=TEST_KEY_ENV,
        transport=transport,
        env={TEST_KEY_ENV: "not-a-real-key"},
        allow_remote=False,
    )


def ollama_tool_response(
    name: Any,
    arguments: Any,
    *,
    content: Any = "",
    extra: dict[str, Any] | None = None,
    calls: list[Any] | None = None,
) -> dict[str, Any]:
    if calls is None:
        entry: dict[str, Any] = {"function": {"name": name, "arguments": arguments}}
        if extra:
            entry.update(extra)
        calls = [entry]
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": content, "tool_calls": calls},
        "done": True,
        "done_reason": "stop",
    }


def openai_tool_response(
    name: Any,
    arguments: Any,
    *,
    content: Any = None,
    extra: dict[str, Any] | None = None,
    calls: list[Any] | None = None,
) -> dict[str, Any]:
    if calls is None:
        entry = {
            "id": "call_1",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        if extra:
            entry.update(extra)
        calls = [entry]
    return {
        "id": "chatcmpl-1",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": calls},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def schema_tool() -> AITool:
    return AITool(
        name="add_numbers",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "first"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )


def request_with_tools() -> AIRequest:
    return AIRequest(messages=(AIMessage.user("2+40?"),), model="test-model", tools=(schema_tool(),))


# -- Ollama ----------------------------------------------------------------


def test_ollama_native_tool_call_is_normalized() -> None:
    transport = FakeTransport(post_result=ollama_tool_response("add_numbers", {"a": 2, "b": 40}))
    response = ollama_provider(transport).generate(request_with_tools())

    assert response.metadata[TOOL_CALL_KEY] == {"name": "add_numbers", "arguments": {"a": 2, "b": 40}}
    assert response.content == ""


def test_ollama_accepts_json_string_arguments() -> None:
    payload = ollama_tool_response("add_numbers", json.dumps({"a": 2, "b": 40}))
    transport = FakeTransport(post_result=payload)
    response = ollama_provider(transport).generate(request_with_tools())

    assert response.metadata[TOOL_CALL_KEY]["arguments"] == {"a": 2, "b": 40}


def test_ollama_plain_answer_carries_no_tool_call() -> None:
    transport = FakeTransport(post_result=ollama_chat_response("42"))
    response = ollama_provider(transport).generate(request_with_tools())

    assert response.content == "42"
    assert TOOL_CALL_KEY not in response.metadata


def test_ollama_malformed_arguments_are_dropped() -> None:
    transport = FakeTransport(
        post_result=ollama_tool_response("add_numbers", "{not json", content="I could not parse")
    )
    response = ollama_provider(transport).generate(request_with_tools())

    assert TOOL_CALL_KEY not in response.metadata
    assert response.content == "I could not parse"


def test_ollama_multiple_tool_calls_are_rejected() -> None:
    calls = [
        {"function": {"name": "add_numbers", "arguments": {"a": 1, "b": 2}}},
        {"function": {"name": "current_time", "arguments": {}}},
    ]
    transport = FakeTransport(post_result=ollama_tool_response("", {}, calls=calls))
    response = ollama_provider(transport).generate(request_with_tools())

    assert TOOL_CALL_KEY not in response.metadata


def test_ollama_tool_call_without_name_is_dropped() -> None:
    transport = FakeTransport(post_result=ollama_tool_response("   ", {"a": 1}))
    response = ollama_provider(transport).generate(request_with_tools())

    assert TOOL_CALL_KEY not in response.metadata


def test_ollama_unsupported_call_type_is_dropped() -> None:
    payload = ollama_tool_response("add_numbers", {"a": 1, "b": 2}, extra={"type": "custom"})
    transport = FakeTransport(post_result=payload)
    response = ollama_provider(transport).generate(request_with_tools())

    assert TOOL_CALL_KEY not in response.metadata


def test_ollama_permission_looking_fields_are_stripped() -> None:
    payload = ollama_tool_response(
        "echo_text", {"text": "hi"}, extra={"approved": True, "bypass": True}
    )
    transport = FakeTransport(post_result=payload)
    response = ollama_provider(transport).generate(request_with_tools())

    call = response.metadata[TOOL_CALL_KEY]
    assert set(call) == {"name", "arguments"}
    assert "approved" not in call and "bypass" not in call


def test_ollama_sends_tool_schemas() -> None:
    transport = FakeTransport(post_result=ollama_chat_response("42"))
    ollama_provider(transport).generate(request_with_tools())

    payload = transport.calls[-1]["payload"]
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "first"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]


def test_ollama_sends_no_tools_when_none_declared() -> None:
    transport = FakeTransport(post_result=ollama_chat_response("42"))
    ollama_provider(transport).generate(AIRequest(messages=(AIMessage.user("hi"),)))

    assert "tools" not in transport.calls[-1]["payload"]


# -- OpenAI-compatible ------------------------------------------------------


def test_openai_native_tool_call_is_normalized() -> None:
    payload = openai_tool_response("add_numbers", json.dumps({"a": 2, "b": 40}))
    transport = FakeTransport(post_result=payload)
    response = openai_provider(transport).generate(request_with_tools())

    assert response.metadata[TOOL_CALL_KEY] == {"name": "add_numbers", "arguments": {"a": 2, "b": 40}}


def test_openai_accepts_object_arguments() -> None:
    transport = FakeTransport(post_result=openai_tool_response("add_numbers", {"a": 2, "b": 40}))
    response = openai_provider(transport).generate(request_with_tools())

    assert response.metadata[TOOL_CALL_KEY]["arguments"] == {"a": 2, "b": 40}


def test_openai_plain_answer_carries_no_tool_call() -> None:
    transport = FakeTransport(post_result=openai_chat_response("42"))
    response = openai_provider(transport).generate(request_with_tools())

    assert response.content == "42"
    assert TOOL_CALL_KEY not in response.metadata


def test_openai_malformed_arguments_are_dropped() -> None:
    payload = openai_tool_response("add_numbers", "not-json", content="cannot parse")
    transport = FakeTransport(post_result=payload)
    response = openai_provider(transport).generate(request_with_tools())

    assert TOOL_CALL_KEY not in response.metadata
    assert response.content == "cannot parse"


def test_openai_multiple_tool_calls_are_rejected() -> None:
    calls = [
        {"id": "1", "type": "function", "function": {"name": "add_numbers", "arguments": "{}"}},
        {"id": "2", "type": "function", "function": {"name": "current_time", "arguments": "{}"}},
    ]
    payload = openai_tool_response("", "{}", calls=calls, content="one thing at a time")
    transport = FakeTransport(post_result=payload)
    response = openai_provider(transport).generate(request_with_tools())

    assert TOOL_CALL_KEY not in response.metadata
    assert response.content == "one thing at a time"


def test_openai_rejected_call_without_text_is_a_provider_error() -> None:
    """An ambiguous call with no text at all is unusable, not an empty success."""
    calls = [
        {"id": "1", "type": "function", "function": {"name": "add_numbers", "arguments": "{}"}},
        {"id": "2", "type": "function", "function": {"name": "current_time", "arguments": "{}"}},
    ]
    transport = FakeTransport(post_result=openai_tool_response("", "{}", calls=calls))

    with pytest.raises(ProviderResponseError):
        openai_provider(transport).generate(request_with_tools())


def test_openai_sends_tool_schemas() -> None:
    transport = FakeTransport(post_result=openai_chat_response("42"))
    openai_provider(transport).generate(request_with_tools())

    payload = transport.calls[-1]["payload"]
    assert payload["tools"][0]["function"]["name"] == "add_numbers"


# -- shared normalisation ---------------------------------------------------


def test_message_without_tool_calls_is_not_a_call() -> None:
    assert tool_call_from_message({"role": "assistant", "content": "hi"}) == (None, None)


def test_non_list_tool_calls_field_is_rejected() -> None:
    call, reason = tool_call_from_message({"tool_calls": {"function": {"name": "x"}}})
    assert call is None
    assert reason


def test_arguments_that_are_not_an_object_are_rejected() -> None:
    call, reason = tool_call_from_message(
        {"tool_calls": [{"function": {"name": "x", "arguments": [1, 2]}}]}
    )
    assert call is None
    assert reason


def test_request_rejects_duplicate_tool_names() -> None:
    with pytest.raises(AIRequestError):
        AIRequest(
            messages=(AIMessage.user("hi"),),
            tools=(AITool(name="add_numbers"), AITool(name="add_numbers")),
        ).validate()


def test_tool_declaration_requires_a_name() -> None:
    with pytest.raises(AIRequestError):
        AITool(name="  ")


def test_tool_declarations_from_registry_hide_permission() -> None:
    from jarvis.core.config import ToolsConfig
    from jarvis.tools import build_tool_system

    tools = build_tool_system(ToolsConfig())
    declarations = tool_declarations(tools.registry)
    names = {tool.name for tool in declarations}

    assert "add_numbers" in names and "echo_text" in names
    for tool in declarations:
        assert "permission" not in tool.as_dict()
        assert tool.parameters["type"] == "object"
