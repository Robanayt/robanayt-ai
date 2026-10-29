"""Cooperative cancellation and turn observation (Phase 6, UI support).

Cancellation is only allowed *between* steps: a tool that is already running is
never interrupted, and the result then says honestly what ran.
"""

from __future__ import annotations

from pathlib import Path

from jarvis.orchestration import TurnCancellation
from jarvis.orchestration.models import DialogStatus
from jarvis.tools.models import ArgumentSpec, PermissionLevel
from tests.orchestration.dialog_fakes import (
    answer,
    make_dialog,
    register_tool,
    structured_call,
)


def test_cancellation_before_the_first_step_executes_nothing(tmp_path: Path) -> None:
    dialog, provider, _memory, tools, _clock = make_dialog(
        tmp_path, [structured_call("side_effect")]
    )
    calls: list[dict[str, object]] = []
    register_tool(
        tools,
        "side_effect",
        lambda arguments, context: calls.append(dict(arguments)) or {"ok": True},
    )

    cancellation = TurnCancellation()
    cancellation.cancel()
    result = dialog.send("сделай", cancellation=cancellation)

    assert result.status is DialogStatus.CANCELLED
    assert calls == []  # nothing was executed
    assert result.tool_calls == ()
    assert provider.requests == []  # not even a model request


def test_cancellation_after_a_tool_ran_reports_the_executed_call(tmp_path: Path) -> None:
    """A started action is not "un-executed" and is not hidden."""
    dialog, _provider, _memory, tools, _clock = make_dialog(
        tmp_path, [structured_call("side_effect"), answer("готово")]
    )
    cancellation = TurnCancellation()
    calls: list[dict[str, object]] = []

    def handler(arguments: dict[str, object], context: object) -> dict[str, object]:
        calls.append(dict(arguments))
        cancellation.cancel()  # the action is already running
        return {"ok": True}

    register_tool(tools, "side_effect", handler)

    result = dialog.send("сделай", cancellation=cancellation)

    assert result.status is DialogStatus.CANCELLED
    assert calls == [{}]  # it really ran
    assert result.tool_call_count == 1  # and it is reported
    assert result.answer == ""  # no answer was invented


def test_cancelled_turn_does_not_approve_a_pending_call(tmp_path: Path) -> None:
    dialog, _provider, _memory, tools, _clock = make_dialog(
        tmp_path, [structured_call("granted_tool", {"text": "hi"})]
    )
    calls: list[dict[str, object]] = []
    register_tool(
        tools,
        "granted_tool",
        lambda arguments, context: calls.append(dict(arguments)) or {"text": "ok"},
        permission=PermissionLevel.APPROVAL,
        arguments=(ArgumentSpec(name="text", kind="string"),),
    )
    tools.policy._known["granted_tool"] = PermissionLevel.APPROVAL

    pending = dialog.send("echo hi").pending
    assert pending is not None

    cancellation = TurnCancellation()
    cancellation.cancel()
    result = dialog.approve_pending(pending.pending_id, cancellation=cancellation)

    assert result.status is DialogStatus.CANCELLED
    assert calls == []  # the approval was not executed
    assert len(dialog.pending()) == 1  # the decision still belongs to the user


def test_observer_reports_real_progress(tmp_path: Path) -> None:
    dialog, _provider, _memory, tools, _clock = make_dialog(
        tmp_path, [structured_call("note_tool"), answer("готово")]
    )
    register_tool(
        tools,
        "note_tool",
        lambda arguments, context: {"text": arguments.get("text", "")},
    )
    events: list[str] = []

    class Recorder:
        def on_model_start(self) -> None:
            events.append("start")

        def on_model_done(self, *, has_tool_call: bool) -> None:
            events.append(f"done:{has_tool_call}")

    result = dialog.send("привет", observer=Recorder())

    assert result.ok
    assert events == ["start", "done:True", "start", "done:False"]


def test_a_failing_observer_never_breaks_the_turn(tmp_path: Path) -> None:
    dialog, _provider, _memory, _tools, _clock = make_dialog(tmp_path, [answer("ok")])

    class Broken:
        def on_model_start(self) -> None:
            raise RuntimeError("observer boom")

        def on_model_done(self, *, has_tool_call: bool) -> None:
            raise RuntimeError("observer boom")

    result = dialog.send("привет", observer=Broken())

    assert result.status is DialogStatus.SUCCESS
    assert result.answer == "ok"
