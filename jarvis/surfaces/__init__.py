"""Desktop surfaces: the UI layer of JARVIS_AI.

Direction of dependency: `jarvis.surfaces` -> `jarvis.core` / `jarvis.ai` /
`jarvis.memory` / `jarvis.tools` / `jarvis.orchestration`. Nothing in the core
imports this package, so the application keeps working unchanged without a UI.

The window itself is built with `tkinter` (standard library), so no new
dependency is added. Importing this package never imports `tkinter`:
availability is exposed as `TK_AVAILABLE`, and the UI modules are imported on
demand by `jarvis.surfaces.ui`.
"""

from __future__ import annotations

from jarvis.surfaces.bridge import DialogBridge
from jarvis.surfaces.protocol import (
    STATE_ANSWERING,
    STATE_ERROR,
    STATE_IDLE,
    STATE_OFFLINE,
    STATE_THINKING,
    STATE_TOOL,
    STATE_WAITING,
    ChatMessage,
    PendingView,
    StateView,
    ToolCallView,
    TurnView,
    pending_view,
    state_view,
    tool_call_view,
    turn_view,
)
from jarvis.surfaces.theme import DEFAULT_THEME, Theme, palette_for_state

try:  # pragma: no cover - depends on the interpreter build
    import tkinter  # noqa: F401

    TK_AVAILABLE = True
except Exception:  # noqa: BLE001 - a missing Tk is a supported state
    TK_AVAILABLE = False

__all__ = [
    "STATE_ANSWERING",
    "STATE_ERROR",
    "STATE_IDLE",
    "STATE_OFFLINE",
    "STATE_THINKING",
    "STATE_TOOL",
    "STATE_WAITING",
    "ChatMessage",
    "DEFAULT_THEME",
    "DialogBridge",
    "PendingView",
    "StateView",
    "TK_AVAILABLE",
    "Theme",
    "ToolCallView",
    "TurnView",
    "palette_for_state",
    "pending_view",
    "state_view",
    "tool_call_view",
    "turn_view",
]
