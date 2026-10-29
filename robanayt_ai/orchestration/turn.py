"""Cooperative turn control: cancellation and progress observation.

A turn can be stopped only *between* its steps - before a model request,
before a tool execution, after a step has completed. A tool that is already
running is never interrupted: stopping is cooperative, so

- no side effect is silently "un-executed";
- the caller never gets a false "nothing happened" report: executed calls stay
  in `DialogResult.tool_calls`, and the status says the turn was cancelled.

Nothing here starts a thread: the flag is set by whoever asked for the stop
(the UI) and read by the loop that owns the turn.
"""

from __future__ import annotations

import threading
from typing import Protocol


class TurnCancellation:
    """One-shot cooperative stop flag for a single turn."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        """Ask the turn to stop at the next safe point."""
        self._event.set()


class TurnObserver(Protocol):
    """Progress callbacks for one turn, called from the turn's own thread.

    Used to report real states only: `on_model_start` when a model request is
    about to be sent, `on_model_done` when the answer arrived. A turn that is
    not doing anything reports nothing.
    """

    def on_model_start(self) -> None: ...

    def on_model_done(self, *, has_tool_call: bool) -> None: ...
