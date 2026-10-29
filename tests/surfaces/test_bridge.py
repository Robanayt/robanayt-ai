"""The bridge is the only path from the UI into the core.

These tests assert that the UI cannot reach past the dialogue layer: the fake
dialogue object exposes nothing but the documented API, and the bridge never
asks for anything else.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from jarvis.core.bus import Event, EventBus
from jarvis.memory.errors import MemoryExportError
from jarvis.orchestration.models import (
    DialogResult,
    DialogStatus,
    PendingApproval,
    ToolCallRecord,
)
from jarvis.surfaces.bridge import DialogBridge
from jarvis.surfaces.protocol import (
    STATE_ANSWERING,
    STATE_CANCELLED,
    STATE_CANCELLING,
    STATE_ERROR,
    STATE_IDLE,
    STATE_THINKING,
    STATE_TOOL,
    STATE_WAITING,
)
from jarvis.tools import TOOL_EXECUTED


class FakeDialog:
    """Only the methods the UI is allowed to call."""

    def __init__(
        self,
        result: DialogResult | None = None,
        pending: tuple[PendingApproval, ...] = (),
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.last_kwargs: dict[str, object] = {}
        self._result = result or DialogResult(
            status=DialogStatus.SUCCESS,
            answer="готово",
            conversation_id="conv_test",
        )
        self._pending = list(pending)

    def send(self, message: str, **kwargs: object) -> DialogResult:
        self.calls.append(("send", message))
        self.last_kwargs = dict(kwargs)
        return self._result

    def approve_pending(self, pending_id: str, **_kwargs: object) -> DialogResult:
        self.calls.append(("approve", pending_id))
        return self._result

    def cancel_pending(self, pending_id: str) -> bool:
        self.calls.append(("cancel", pending_id))
        if self._pending:
            self._pending.pop(0)
            return True
        return False

    def pending(self) -> tuple[PendingApproval, ...]:
        return tuple(self._pending)

    def describe(self) -> dict[str, object]:
        return {
            "enabled": True,
            "max_tool_calls_per_message": 3,
            "allow_text_tool_calls": False,
            "pending": len(self._pending),
        }

    def close(self) -> None:
        self.calls.append(("close", ""))


class FailingDialog(FakeDialog):
    def send(self, message: str, **_kwargs: object) -> DialogResult:
        raise RuntimeError("boom")


def make_bridge(dialog: FakeDialog, **kwargs: object) -> DialogBridge:
    return DialogBridge(dialog, background=False, **kwargs)  # type: ignore[arg-type]


def test_send_records_user_and_assistant_messages() -> None:
    bridge = make_bridge(FakeDialog())

    assert bridge.send("привет") is True

    view = bridge.snapshot()
    assert [message.role for message in view.messages] == ["user", "assistant"]
    assert view.messages[1].text == "готово"
    assert view.conversation_id == "conv_test"
    assert view.state == STATE_IDLE


def test_empty_input_is_refused() -> None:
    dialog = FakeDialog()
    bridge = make_bridge(dialog)

    assert bridge.send("   ") is False
    assert bridge.snapshot().messages == ()
    assert dialog.calls == []


def test_pending_call_waits_and_shows_its_arguments() -> None:
    pending = PendingApproval(
        pending_id="dlg_1",
        tool="write_file",
        version="1.0.0",
        arguments={"path": "notes.txt"},
        arguments_digest="digest-1",
        reason="запись файла",
    )
    result = DialogResult(
        status=DialogStatus.APPROVAL_REQUIRED,
        pending=pending,
        conversation_id="conv_test",
    )
    bridge = make_bridge(FakeDialog(result=result, pending=(pending,)))

    bridge.send("запиши файл")

    view = bridge.snapshot()
    assert view.state == STATE_WAITING
    assert view.pending[0].tool == "write_file"
    # Arguments are visible: consent has to be an informed decision.
    assert view.pending[0].arguments == {"path": "notes.txt"}


def test_approval_goes_through_the_dialogue_layer() -> None:
    dialog = FakeDialog()
    bridge = make_bridge(dialog)

    assert bridge.approve("dlg_1") is True
    assert ("approve", "dlg_1") in dialog.calls


def test_cancel_reports_whether_the_call_was_dropped() -> None:
    pending = PendingApproval(pending_id="dlg_1", tool="write_file", version="1.0.0")
    bridge = make_bridge(FakeDialog(pending=(pending,)))

    assert bridge.cancel("dlg_1") is True
    assert bridge.cancel("dlg_1") is False


def test_failure_is_reported_and_not_masked() -> None:
    bridge = make_bridge(FailingDialog())

    bridge.send("привет")

    view = bridge.snapshot()
    assert view.state == STATE_ERROR
    assert "boom" in (view.last_error or "")
    events = bridge.poll()
    assert events[-1].kind == "error"


def test_bus_events_mark_real_tool_activity() -> None:
    bus = EventBus()
    bridge = make_bridge(FakeDialog(), bus=bus)

    bus.publish(Event(TOOL_EXECUTED, {"status": "success"}))

    assert bridge.snapshot().state == STATE_TOOL


def test_new_conversation_clears_the_transcript() -> None:
    bridge = make_bridge(FakeDialog())
    bridge.send("привет")

    bridge.new_conversation()

    view = bridge.snapshot()
    assert view.messages == ()
    assert view.conversation_id is None


def test_tool_records_reach_the_view() -> None:
    record = ToolCallRecord(
        tool="add_numbers", status="success", sequence=1, execution_id="exec_1"
    )
    result = DialogResult(
        status=DialogStatus.SUCCESS,
        answer="42",
        conversation_id="conv_test",
        tool_calls=(record,),
    )

    bridge = make_bridge(FakeDialog(result=result))
    bridge.send("сколько будет 40 + 2")

    events = bridge.poll()
    assert events[-1].turn is not None
    assert events[-1].turn.tool_calls[0].tool == "add_numbers"


# -- cancellation -----------------------------------------------------------


class WaitingDialog(FakeDialog):
    """A turn that keeps running until it is asked to stop."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def send(self, message: str, **kwargs: object) -> DialogResult:
        self.calls.append(("send", message))
        self.last_kwargs = dict(kwargs)
        cancellation = kwargs.get("cancellation")
        self.started.set()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if cancellation is not None and getattr(cancellation, "requested", False):
                return DialogResult(
                    status=DialogStatus.CANCELLED,
                    error="the turn was cancelled",
                    error_type="TurnCancelled",
                    conversation_id="conv_test",
                )
            time.sleep(0.005)
        return DialogResult(
            status=DialogStatus.SUCCESS, answer="ok", conversation_id="conv_test"
        )


def _wait_for_state(bridge: DialogBridge, state: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        bridge.poll()
        if bridge.snapshot().state == state:
            return True
        time.sleep(0.01)
    return False


def test_cancel_turn_stops_a_running_turn() -> None:
    dialog = WaitingDialog()
    bridge = DialogBridge(dialog, background=True)  # a real worker thread

    bridge.send("привет")
    assert dialog.started.wait(10.0)

    assert bridge.cancel_turn() is True
    assert _wait_for_state(bridge, STATE_CANCELLED) is True

    view = bridge.snapshot()
    assert view.state == STATE_CANCELLED
    assert any("отменён" in message.text for message in view.messages)
    bridge.close()


def test_cancelling_when_nothing_runs_is_refused() -> None:
    bridge = make_bridge(FakeDialog())

    assert bridge.cancel_turn() is False
    assert bridge.snapshot().state == STATE_IDLE


def test_a_pending_call_outlives_a_cancellation_request() -> None:
    """Cancelling the turn is not a decision on a pending call."""
    pending = PendingApproval(pending_id="dlg_1", tool="write_file", version="1.0.0")
    dialog = FakeDialog(
        result=DialogResult(
            status=DialogStatus.APPROVAL_REQUIRED,
            pending=pending,
            conversation_id="conv_test",
        ),
        pending=(pending,),
    )
    bridge = make_bridge(dialog)

    bridge.send("запиши файл")
    assert bridge.snapshot().state == STATE_WAITING

    assert bridge.cancel_turn() is False  # nothing is running: it waits for a decision
    assert len(bridge.snapshot().pending) == 1


def test_the_ui_thread_is_not_blocked_by_the_turn() -> None:
    dialog = WaitingDialog()
    bridge = DialogBridge(dialog, background=True)

    bridge.send("привет")
    assert dialog.started.wait(10.0)

    started = time.time()
    view = bridge.snapshot()  # what the UI thread does every frame
    elapsed = time.time() - started

    assert view.state in {STATE_THINKING, STATE_CANCELLING}
    assert elapsed < 0.5  # the interface never waits for the core

    assert bridge.cancel_turn() is True
    assert _wait_for_state(bridge, STATE_CANCELLED) is True
    bridge.close()


def test_progress_observation_sets_real_states() -> None:
    bridge = make_bridge(FakeDialog())

    bridge.on_model_start()
    assert bridge.snapshot().state == STATE_THINKING

    bridge.on_model_done(has_tool_call=True)
    assert bridge.snapshot().state == STATE_THINKING

    bridge.on_model_done(has_tool_call=False)
    assert bridge.snapshot().state == STATE_ANSWERING


# -- history and export -----------------------------------------------------


@dataclass(frozen=True)
class Conversation:
    """The shape `MemoryService` returns: objects with `id` and `title`."""

    id: str
    title: str = ""


class FakeConversations:
    """Slice-based conversation list, so paging is observable."""

    def __init__(self, items: tuple[Conversation, ...]) -> None:
        self.items = list(items)
        self.calls: list[tuple[int, int]] = []

    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> tuple:
        self.calls.append((limit, offset))
        return tuple(self.items[offset : offset + limit])

    def require_conversation(self, conversation_id: str) -> Conversation:
        return next(item for item in self.items if item.id == conversation_id)

    def get_messages(self, conversation_id: str) -> tuple:
        return ()


class FakeMemory:
    def __init__(
        self,
        directory: Path,
        conversations: tuple[Conversation, ...] = (),
        *,
        fail: bool = False,
    ) -> None:
        self.config = SimpleNamespace(export_directory=str(directory))
        self.conversations = FakeConversations(conversations)
        self.fail = fail
        self.export_calls: list[tuple[str, str, bool]] = []
        self.enabled = True
        self.database_path = directory / "jarvis.db"

    def export_conversation_to_file(
        self, conversation_id: str, path: object, *, overwrite: bool = True
    ) -> Path:
        self.export_calls.append((conversation_id, str(path), overwrite))
        if self.fail:
            raise MemoryExportError("disk is full")
        target = Path(str(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        return target


def test_empty_history_is_empty() -> None:
    bridge = make_bridge(FakeDialog(), memory=FakeMemory(Path(".")))

    assert bridge.conversations() == ()


def test_history_is_loaded_in_pages(tmp_path: Path) -> None:
    items = tuple(Conversation(f"conv_{index}", f"Диалог {index}") for index in range(5))
    expected = tuple((item.id, item.title) for item in items)
    memory = FakeMemory(tmp_path, items)
    bridge = make_bridge(FakeDialog(), memory=memory)

    assert bridge.conversations(limit=2, offset=0) == expected[:2]
    assert bridge.conversations(limit=2, offset=2) == expected[2:4]
    assert bridge.conversations(limit=2, offset=4) == expected[4:]
    assert bridge.conversations(limit=2, offset=6) == ()
    assert memory.conversations.calls[0] == (2, 0)


def test_opening_a_conversation_keeps_the_id_flow() -> None:
    items = (Conversation("conv_1", "Первый"), Conversation("conv_2", "Второй"))
    dialog = FakeDialog()
    bridge = make_bridge(dialog, memory=FakeMemory(Path("."), items))

    assert bridge.open_conversation("conv_2") is True
    assert bridge.snapshot().conversation_id == "conv_2"

    bridge.send("продолжаем")
    assert dialog.last_kwargs["conversation_id"] == "conv_2"


def test_opening_an_unknown_conversation_is_refused() -> None:
    bridge = make_bridge(
        FakeDialog(), memory=FakeMemory(Path("."), (Conversation("conv_1", "x"),))
    )

    assert bridge.open_conversation("conv_missing") is False


def test_export_uses_the_memory_layer(tmp_path: Path) -> None:
    memory = FakeMemory(tmp_path)
    bridge = make_bridge(FakeDialog(), memory=memory)
    bridge.send("привет")
    target = tmp_path / "exports" / "dialog.json"

    ok, message = bridge.export_conversation(None, target)

    assert ok is True
    assert Path(message) == target
    assert target.is_file()
    assert memory.export_calls[0][0] == "conv_test"


def test_export_reports_an_error_instead_of_raising(tmp_path: Path) -> None:
    memory = FakeMemory(tmp_path, fail=True)
    bridge = make_bridge(FakeDialog(), memory=memory)
    bridge.send("привет")

    ok, message = bridge.export_conversation(None, tmp_path / "out.json")

    assert ok is False
    assert "disk is full" in message


def test_export_without_a_conversation_is_explained() -> None:
    bridge = make_bridge(FakeDialog(), memory=FakeMemory(Path(".")))

    ok, message = bridge.export_conversation(None, "out.json")

    assert ok is False
    assert "нет диалога" in message


def test_export_without_memory_is_explained() -> None:
    bridge = make_bridge(FakeDialog())

    ok, message = bridge.export_conversation("conv_1", "out.json")

    assert ok is False
    assert "память отключена" in message
