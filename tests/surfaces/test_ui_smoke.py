"""Window smoke tests.

The widgets are really built (so a broken layout fails the suite) but never
shown: the window is created withdrawn and destroyed immediately.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest

pytest.importorskip("tkinter")

from robanayt_ai.surfaces.bridge import DialogBridge  # noqa: E402
from robanayt_ai.surfaces.protocol import (  # noqa: E402
    STATE_ERROR,
    STATE_IDLE,
    STATE_THINKING,
    STATE_TOOL,
    STATE_WAITING,
    PendingView,
)
from robanayt_ai.surfaces.theme import (  # noqa: E402
    DEFAULT_THEME,
    REDUCED_MOTION_THEME,
    palette_for_state,
)
from robanayt_ai.surfaces.ui.approval import ApprovalDialog  # noqa: E402
from robanayt_ai.surfaces.ui.core_widget import CoreWidget  # noqa: E402
from robanayt_ai.surfaces.ui.window import HolographicApp  # noqa: E402
from robanayt_ai.orchestration.models import DialogResult, DialogStatus  # noqa: E402
from tests.surfaces.test_bridge import (  # noqa: E402
    Conversation,
    FakeDialog,
    FakeMemory,
)


def test_window_builds_and_closes() -> None:
    bridge = DialogBridge(FakeDialog(), background=False)
    app = HolographicApp(bridge, hidden=True)

    try:
        app.root.update_idletasks()
        assert app.root.title() == "Robanayt AI"
    finally:
        app.close()


def test_window_survives_a_state_switch() -> None:
    bridge = DialogBridge(FakeDialog(), background=False)
    app = HolographicApp(bridge, hidden=True)

    try:
        app._set_motion(True)
        app.root.update_idletasks()
        app._set_motion(False)
        app.root.update_idletasks()
    finally:
        app.close()


def test_cancel_is_offered_only_while_a_turn_runs() -> None:
    bridge = DialogBridge(FakeDialog(), background=False)
    app = HolographicApp(bridge, hidden=True)

    try:
        app.root.update_idletasks()
        assert str(app._cancel_button["state"]) == "disabled"
    finally:
        app.close()


def test_cancelled_turn_is_shown_unambiguously() -> None:
    dialog = FakeDialog(
        result=DialogResult(
            status=DialogStatus.CANCELLED,
            error="the turn was cancelled",
            error_type="TurnCancelled",
            conversation_id="conv_test",
        )
    )
    bridge = DialogBridge(dialog, background=False)
    app = HolographicApp(bridge, hidden=True)

    try:
        bridge.send("привет")
        app._poll()  # one frame, as the main loop would

        assert app._status_var.get() == "ход отменён"
        assert "отменён" in app._transcript.get("1.0", "end")
        assert str(app._cancel_button["state"]) == "disabled"
    finally:
        app.close()


def test_history_is_paged_in_the_sidebar(tmp_path: "Path") -> None:
    items = tuple(Conversation(f"conv_{index}", f"Диалог {index}") for index in range(5))
    bridge = DialogBridge(
        FakeDialog(), background=False, memory=FakeMemory(tmp_path, items)
    )
    app = HolographicApp(bridge, hidden=True)

    try:
        app._page_size = 2
        app._refresh_history(reset=True)
        assert app._history.size() == 2
        assert str(app._more_button["state"]) == "normal"

        app._load_more_history()
        assert app._history.size() == 4

        app._load_more_history()
        assert app._history.size() == 5
        assert str(app._more_button["state"]) == "disabled"  # history ended
    finally:
        app.close()


def test_core_draws_the_cancellation_states() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        widget = CoreWidget(root, DEFAULT_THEME)
        widget.pack()
        for state in ("cancelling", "cancelled"):
            widget.set_state(state)
            widget._draw()
        root.update_idletasks()
        widget.stop()
    finally:
        root.destroy()


def test_core_widget_draws_every_state() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        widget = CoreWidget(root, DEFAULT_THEME)
        widget.pack()
        for state in (
            STATE_IDLE,
            STATE_THINKING,
            STATE_TOOL,
            STATE_WAITING,
            STATE_ERROR,
            "offline",
        ):
            widget.set_state(state)
            widget._draw()
        widget.set_theme(REDUCED_MOTION_THEME)
        widget._draw()
        root.update_idletasks()

        assert widget.winfo_reqwidth() >= DEFAULT_THEME.metrics.core
        widget.stop()
    finally:
        root.destroy()


def test_every_state_has_its_own_colour() -> None:
    colours = {
        palette_for_state(STATE_IDLE),
        palette_for_state(STATE_WAITING),
        palette_for_state(STATE_ERROR),
    }

    assert len(colours) == 3


def test_approval_dialog_shows_the_arguments() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        pending = PendingView(
            pending_id="dlg_1",
            tool="write_file",
            version="1.0.0",
            arguments={"path": "notes.txt"},
            reason="запись файла",
        )
        dialog = ApprovalDialog(root, pending, DEFAULT_THEME)
        texts: list[str] = []

        def collect(widget: tk.Misc) -> None:
            if isinstance(widget, tk.Text):
                texts.append(widget.get("1.0", "end"))
            for child in widget.winfo_children():
                collect(child)

        collect(dialog._top)
        assert any("notes.txt" in text for text in texts)
        dialog._top.destroy()
    finally:
        root.destroy()
