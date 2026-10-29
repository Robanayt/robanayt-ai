"""Explicit application lifecycle.

States are explicit and transitions are validated: an illegal transition
raises `InvalidTransitionError` instead of being silently accepted. Every
transition is logged and reported to an optional callback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

TransitionListener = Callable[["State", "State"], None]


class State(str, Enum):
    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


ALLOWED_TRANSITIONS: dict[State, frozenset[State]] = {
    State.NEW: frozenset({State.STARTING, State.FAILED}),
    State.STARTING: frozenset({State.RUNNING, State.STOPPING, State.FAILED}),
    State.RUNNING: frozenset({State.STOPPING, State.FAILED}),
    State.STOPPING: frozenset({State.STOPPED, State.FAILED}),
    State.STOPPED: frozenset({State.NEW}),
    State.FAILED: frozenset({State.STOPPING, State.STOPPED, State.NEW}),
}

TERMINAL_STATES: frozenset[State] = frozenset({State.STOPPED, State.FAILED})


class InvalidTransitionError(RuntimeError):
    """Raised when a state change is not allowed."""

    def __init__(self, current: State, target: State) -> None:
        super().__init__(
            f"invalid lifecycle transition: {current.value} -> {target.value}"
        )
        self.current = current
        self.target = target


class Lifecycle:
    """Tracks startup/shutdown state; usable as a context manager."""

    def __init__(self, on_transition: TransitionListener | None = None) -> None:
        self._state = State.NEW
        self._failure_reason: str | None = None
        self._listeners: list[TransitionListener] = []
        if on_transition is not None:
            self._listeners.append(on_transition)

    @property
    def state(self) -> State:
        return self._state

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def is_running(self) -> bool:
        return self._state is State.RUNNING

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    def add_listener(self, listener: TransitionListener) -> None:
        self._listeners.append(listener)

    def can_transition_to(self, target: State) -> bool:
        return target in ALLOWED_TRANSITIONS[self._state]

    def transition_to(self, target: State) -> State:
        if not self.can_transition_to(target):
            error = InvalidTransitionError(self._state, target)
            logger.error("%s", error)
            raise error
        previous, self._state = self._state, target
        if target in {State.NEW, State.STARTING}:
            # A fresh start clears the previous failure; shutting down after a
            # failure keeps the reason available for diagnostics.
            self._failure_reason = None
        logger.info("lifecycle: %s -> %s", previous.value, target.value)
        for listener in self._listeners:
            listener(previous, target)
        return target

    def start(self) -> State:
        return self.transition_to(State.STARTING)

    def mark_running(self) -> State:
        return self.transition_to(State.RUNNING)

    def begin_stop(self) -> State:
        return self.transition_to(State.STOPPING)

    def mark_stopped(self) -> State:
        return self.transition_to(State.STOPPED)

    def fail(self, reason: str) -> State:
        """Mark a failed startup or shutdown. Never raises on legal states."""
        self._failure_reason = reason
        logger.error("lifecycle failure: %s", reason)
        return self.transition_to(State.FAILED)

    def reset(self) -> State:
        """Back to NEW - only allowed from a terminal state (tests, restarts)."""
        return self.transition_to(State.NEW)

    def shutdown_cleanly(self) -> State:
        """Best-effort stop sequence used by the application teardown."""
        if self._state in {State.STARTING, State.RUNNING, State.FAILED}:
            self.begin_stop()
        if self._state is State.STOPPING:
            self.mark_stopped()
        return self._state

    def __enter__(self) -> Lifecycle:
        self.start()
        self.mark_running()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            if not self.is_terminal:
                self.fail(str(exc) or exc_type.__name__)
            self.shutdown_cleanly()
            return
        self.shutdown_cleanly()
