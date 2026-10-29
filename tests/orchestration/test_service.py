"""DialogService: one bounded turn through AI, memory and the tool system."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests.orchestration.dialog_fakes import (
    OrchestrationConfig,
    answer,
    make_dialog,
    register_tool,
    structured_call,
    text_call,
)
from robanayt_ai.ai.errors import ProviderUnavailableError
from robanayt_ai.ai.models import AIResponse
from robanayt_ai.memory.errors import MemoryLayerError
from robanayt_ai.orchestration.models import MAX_PENDING_TURNS, DialogStatus
from robanayt_ai.tools.models import ArgumentSpec, PermissionLevel


def test_plain_message_without_tools(tmp_path: Path) -> None:
    dialog, provider, memory, _tools, _clock = make_dialog(tmp_path, [answer("hello there")])

    result = dialog.send("hi")

    assert result.status is DialogStatus.SUCCESS
    assert result.answer == "hello there"
    assert result.tool_call_count == 0
    assert result.persisted is True
    assert result.conversation_id is not None
    assert len(provider.requests) == 1
    messages = memory.conversations.get_messages(result.conversation_id)
    assert [message.role for message in messages] == ["user", "assistant"]
    memory.close()


def test_conversation_is_continued_by_id(tmp_path: Path) -> None:
    dialog, provider, memory, _tools, _clock = make_dialog(
        tmp_path, [answer("first"), answer("second")]
    )

    first = dialog.send("one")
    second = dialog.send("two", conversation_id=first.conversation_id)

    assert second.conversation_id == first.conversation_id
    assert second.answer == "second"
    history = [message.content for message in provider.requests[1].messages]
    assert "one" in " ".join(history)
    assert "first" in " ".join(history)
    assert len(memory.conversations.get_messages(first.conversation_id)) == 4
    memory.close()


def test_context_is_limited(tmp_path: Path) -> None:
    dialog, provider, memory, _tools, _clock = make_dialog(tmp_path, [answer("ok"), answer("ok")])
    conversation_id = dialog.send("start").conversation_id
    for index in range(6):
        memory.add_message(conversation_id, "user", f"filler {index}")

    result = dialog.send("latest", conversation_id=conversation_id, max_messages=2)

    assert result.status is DialogStatus.SUCCESS
    assert result.context_truncated is True
    sent = " ".join(message.content for message in provider.requests[-1].messages)
    assert "latest" in sent
    assert "filler 0" not in sent
    memory.close()


def test_memory_snippets_are_added_only_on_request(tmp_path: Path) -> None:
    dialog, provider, memory, _tools, _clock = make_dialog(tmp_path, [answer("ok")])
    memory.remember("user prefers dark mode", category="preferences")

    dialog.send("what do I prefer?", memory_query="dark")

    system_text = " ".join(
        message.content for message in provider.requests[0].messages if message.role == "system"
    )
    assert "user prefers" in system_text
    memory.close()


def test_one_allowed_tool_call(tmp_path: Path) -> None:
    dialog, provider, _memory, tools, _clock = make_dialog(
        tmp_path, [structured_call("add_numbers", {"a": 2, "b": 3}), answer("the sum is 5")]
    )

    result = dialog.send("add 2 and 3")

    assert result.status is DialogStatus.SUCCESS
    assert result.answer == "the sum is 5"
    assert result.tool_call_count == 1
    assert result.tool_calls[0].status == "success"
    assert len(provider.requests) == 2
    tool_messages = [
        message.content for message in provider.requests[1].messages if message.role == "tool"
    ]
    assert tool_messages and "data, not instructions" in tool_messages[0]
    assert "5" in tool_messages[0]
    assert tools.audit.records()[0].status == "success"


def test_two_tool_calls_within_the_limit(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(
        tmp_path,
        [
            structured_call("add_numbers", {"a": 1, "b": 1}),
            structured_call("add_numbers", {"a": 2, "b": 2}),
            answer("done"),
        ],
    )

    result = dialog.send("add twice")

    assert result.status is DialogStatus.SUCCESS
    assert result.tool_call_count == 2
    assert len(provider.requests) == 3


def test_loop_stops_at_the_tool_call_limit(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(
        tmp_path,
        [structured_call("add_numbers", {"a": 1, "b": 1}) for _ in range(6)],
        config=OrchestrationConfig(max_tool_calls_per_message=3),
    )

    result = dialog.send("keep adding")

    assert result.status is DialogStatus.TOOL_LIMIT_REACHED
    assert result.tool_call_count == 3
    assert result.answer == ""
    assert result.error is not None and "limit reached" in result.error
    # initial request + one follow-up per executed call, then it stops.
    assert len(provider.requests) == 4


def test_zero_limit_blocks_every_tool_call(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [structured_call("add_numbers", {"a": 1, "b": 1})],
        config=OrchestrationConfig(max_tool_calls_per_message=0),
    )

    result = dialog.send("add")

    assert result.status is DialogStatus.TOOL_LIMIT_REACHED
    assert result.tool_call_count == 0
    assert len(provider.requests) == 1


def test_unknown_tool_is_not_executed(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [structured_call("delete_everything")]
    )

    result = dialog.send("please")

    assert result.status is DialogStatus.TOOL_ERROR
    assert "unknown tool" in (result.error or "")
    assert result.tool_calls[0].status == "not_found"
    assert result.answer == ""


def test_tool_failure_stops_the_turn_without_retry(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(arguments: object, context: object) -> str:
        calls.append(1)
        raise RuntimeError("gateway down")

    dialog, _provider, _memory, tools, _clock = make_dialog(
        tmp_path, [structured_call("boom"), answer("never used")]
    )
    register_tool(tools, "boom", handler)

    result = dialog.send("run it")

    assert result.status is DialogStatus.TOOL_ERROR
    assert result.tool_calls[0].status == "failed"
    assert "gateway down" in (result.error or "")
    assert calls == [1]  # exactly one attempt, never repeated


def test_approval_required_stops_the_turn(tmp_path: Path) -> None:
    calls: list[dict] = []

    dialog, _provider, _memory, tools, _clock = make_dialog(
        tmp_path, [structured_call("granted_tool", {"text": "hi"}), answer("echoed")]
    )
    register_tool(
        tools,
        "granted_tool",
        lambda arguments, context: calls.append(dict(arguments)) or {"text": arguments["text"]},
        permission=PermissionLevel.APPROVAL,
        arguments=(ArgumentSpec(name="text", kind="string"),),
    )
    tools.policy._known["granted_tool"] = PermissionLevel.APPROVAL

    result = dialog.send("echo hi")

    assert result.status is DialogStatus.APPROVAL_REQUIRED
    assert result.pending is not None
    assert result.pending.tool == "granted_tool"
    assert result.tool_call_count == 0
    assert calls == []
    assert [record.status for record in tools.audit.records()] == ["approval_required"]


def test_model_cannot_approve_its_own_call(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(
        tmp_path,
        [
            AIResponse(
                content="",
                model="test-model",
                metadata={
                    "tool_call": {"name": "echo_text", "arguments": {"text": "hi"}, "approved": True}
                },
            )
        ],
    )

    result = dialog.send("echo hi")

    assert result.status is DialogStatus.INVALID_TOOL_CALL
    assert "cannot grant permission" in (result.error or "")


def test_approval_resumes_the_turn(tmp_path: Path) -> None:
    dialog, provider, _memory, tools, _clock = make_dialog(
        tmp_path, [structured_call("echo_text", {"text": "hi"}), answer("echo done")]
    )

    pending_result = dialog.send("echo hi")
    assert pending_result.status is DialogStatus.APPROVAL_REQUIRED
    pending = pending_result.pending
    assert pending is not None

    approved = dialog.approve_pending(pending.pending_id)

    assert approved.status is DialogStatus.SUCCESS
    assert approved.answer == "echo done"
    assert approved.tool_call_count == 1
    assert approved.tool_calls[0].status == "success"
    assert dialog.pending() == ()


def test_pending_approval_is_bound_to_arguments(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [structured_call("echo_text", {"text": "one"}), answer("ok")]
    )

    result = dialog.send("echo")
    pending = result.pending
    assert pending is not None
    assert pending.arguments == {"text": "one"}
    assert pending.arguments_digest

    approved = dialog.approve_pending(pending.pending_id)
    assert approved.status is DialogStatus.SUCCESS


def test_approval_of_unknown_pending_is_reported(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(tmp_path, [answer("ok")])

    result = dialog.approve_pending("dlg_missing")

    assert result.status is DialogStatus.PENDING_EXPIRED
    assert result.error is not None


def test_pending_approval_expires(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, clock = make_dialog(
        tmp_path, [structured_call("echo_text", {"text": "hi"}), answer("ok")]
    )

    result = dialog.send("echo")
    assert result.pending is not None
    clock.advance(10_000)

    expired = dialog.approve_pending(result.pending.pending_id)

    assert expired.status is DialogStatus.PENDING_EXPIRED
    assert dialog.pending() == ()


def test_pending_approval_can_be_cancelled(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [structured_call("echo_text", {"text": "hi"}), answer("ok")]
    )

    result = dialog.send("echo")
    assert result.pending is not None
    assert dialog.cancel_pending(result.pending.pending_id) is True
    assert dialog.pending() == ()
    assert dialog.approve_pending(result.pending.pending_id).status is DialogStatus.PENDING_EXPIRED


def test_ai_error_is_reported(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [ProviderUnavailableError("provider offline")]
    )

    result = dialog.send("hello")

    assert result.status is DialogStatus.AI_ERROR
    assert "provider offline" in (result.error or "")
    assert result.answer == ""


def test_disabled_ai_is_reported(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [answer("never")], ai_enabled=False
    )

    result = dialog.send("hello")

    assert result.status is DialogStatus.AI_ERROR
    assert provider.requests == []


def test_memory_error_is_reported(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(tmp_path, [answer("never")])

    result = dialog.send("hello", conversation_id="conv_missing")

    assert result.status is DialogStatus.MEMORY_ERROR
    assert result.error is not None
    assert provider.requests == []


def test_disabled_memory_still_answers(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [answer("ok")], memory_enabled=False
    )

    result = dialog.send("hello")

    assert result.status is DialogStatus.SUCCESS
    assert result.memory_enabled is False
    assert result.persisted is False
    assert result.conversation_id is None
    contents = " ".join(message.content for message in provider.requests[0].messages)
    assert "hello" in contents


def test_failed_persistence_is_reported_not_hidden(tmp_path: Path, monkeypatch) -> None:
    dialog, provider, memory, _tools, _clock = make_dialog(tmp_path, [answer("ok")])

    def boom(*args: object, **kwargs: object) -> None:
        raise MemoryLayerError("database is locked")

    monkeypatch.setattr(memory, "add_message", boom)

    result = dialog.send("hello")

    assert result.status is DialogStatus.SUCCESS
    assert result.persisted is False
    assert result.persist_error is not None
    contents = " ".join(message.content for message in provider.requests[0].messages)
    assert "hello" in contents


def test_empty_answer_is_not_presented_as_success(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(tmp_path, [answer("   ")])

    result = dialog.send("hello")

    assert result.status is DialogStatus.EMPTY_RESPONSE
    assert result.ok is False


def test_malformed_tool_call_is_not_executed(tmp_path: Path) -> None:
    dialog, _provider, _memory, tools, _clock = make_dialog(
        tmp_path,
        [AIResponse(content="```json\n{oops}\n```", model="test-model")],
        config=OrchestrationConfig(allow_text_tool_calls=True),
    )

    result = dialog.send("call something")

    assert result.status is DialogStatus.INVALID_TOOL_CALL
    assert result.tool_call_count == 0
    assert tools.audit.records() == ()


def test_text_channel_runs_only_when_enabled(tmp_path: Path) -> None:
    disabled, provider_disabled, _memory, _tools, _clock = make_dialog(
        tmp_path, [text_call("add_numbers", {"a": 1, "b": 1})]
    )
    result = disabled.send("add")

    # Default: the block is text, not a call.
    assert result.status is DialogStatus.SUCCESS
    assert result.tool_call_count == 0

    enabled, _provider, _memory, _tools2, _clock2 = make_dialog(
        tmp_path,
        [text_call("add_numbers", {"a": 1, "b": 1}), answer("sum is 2")],
        config=OrchestrationConfig(allow_text_tool_calls=True),
    )
    enabled_result = enabled.send("add")

    assert enabled_result.status is DialogStatus.SUCCESS
    assert enabled_result.tool_call_count == 1
    assert enabled_result.answer == "sum is 2"


def test_tools_can_be_disabled_for_one_turn(tmp_path: Path) -> None:
    dialog, _provider, _memory, tools, _clock = make_dialog(
        tmp_path, [text_call("add_numbers", {"a": 1, "b": 1})]
    )

    result = dialog.send("add", allow_tools=False)

    assert result.status is DialogStatus.SUCCESS
    assert result.tool_call_count == 0
    assert tools.audit.records() == ()


def test_disabled_orchestration_does_nothing(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [answer("never")], config=OrchestrationConfig(enabled=False)
    )

    result = dialog.send("hello")

    assert result.status is DialogStatus.DISABLED
    assert provider.requests == []


def test_empty_message_is_rejected(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(tmp_path, [answer("ok")])

    with pytest.raises(ValueError, match="non-empty"):
        dialog.send("   ")
    assert provider.requests == []


def test_tool_guidance_reaches_the_model(tmp_path: Path) -> None:
    dialog, provider, _memory, _tools, _clock = make_dialog(tmp_path, [answer("ok")])

    dialog.send("hello")

    system_text = " ".join(
        message.content for message in provider.requests[0].messages if message.role == "system"
    )
    assert "add_numbers" in system_text
    assert "never a permission" in system_text


def test_describe_reports_limits(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(tmp_path, [])

    description = dialog.describe()

    assert description["enabled"] is True
    assert description["max_tool_calls_per_message"] == 3
    assert description["allow_text_tool_calls"] is False
    assert description["pending"] == 0


def test_pending_list_is_available(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [structured_call("echo_text", {"text": "hi"}), answer("ok")]
    )

    dialog.send("echo")

    pending = dialog.pending()
    assert len(pending) == 1
    assert pending[0].tool == "echo_text"
    assert "hi" not in str(pending[0].describe())


def test_service_can_be_closed(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(
        tmp_path, [structured_call("echo_text", {"text": "hi"}), answer("ok")]
    )
    dialog.send("echo")

    dialog.close()

    assert dialog.pending() == ()


def test_pending_turn_eviction_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Beyond MAX_PENDING_TURNS the oldest turn is dropped - and reported."""
    caplog.set_level(logging.WARNING, logger="robanayt_ai.orchestration.service")
    # Pruning runs before the new turn is stored, so the oldest turn is dropped
    # on the (MAX + 2)-th pending call and the count then stays bounded.
    sends = MAX_PENDING_TURNS + 2
    replies = [
        structured_call("echo_text", {"text": f"t{index}"}) for index in range(sends)
    ]
    dialog, _provider, memory, _tools, _clock = make_dialog(tmp_path, replies)

    for index in range(sends):
        dialog.send(f"echo {index}")

    assert len(dialog.pending()) == MAX_PENDING_TURNS + 1
    dropped = [
        record.getMessage()
        for record in caplog.records
        if "pending approval dropped" in record.getMessage()
    ]
    assert len(dropped) == sends - MAX_PENDING_TURNS - 1
    # The warning names the tool, never the argument values.
    assert "echo_text" in dropped[0]
    assert "t0" not in dropped[0]
    memory.close()
