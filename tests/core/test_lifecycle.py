"""Lifecycle: explicit states, validated transitions, clean shutdown."""

from __future__ import annotations

import pytest

from jarvis.core.lifecycle import InvalidTransitionError, Lifecycle, State


def test_initial_state_is_new() -> None:
    lifecycle = Lifecycle()

    assert lifecycle.state is State.NEW
    assert lifecycle.is_running is False
    assert lifecycle.failure_reason is None


def test_startup_sequence() -> None:
    lifecycle = Lifecycle()

    assert lifecycle.start() is State.STARTING
    assert lifecycle.mark_running() is State.RUNNING
    assert lifecycle.is_running is True


def test_invalid_transition_raises() -> None:
    lifecycle = Lifecycle()

    with pytest.raises(InvalidTransitionError, match="new -> running"):
        lifecycle.mark_running()

    assert lifecycle.state is State.NEW


def test_cannot_start_twice() -> None:
    lifecycle = Lifecycle()
    lifecycle.start()

    with pytest.raises(InvalidTransitionError):
        lifecycle.start()


def test_full_cycle_ends_stopped() -> None:
    lifecycle = Lifecycle()
    lifecycle.start()
    lifecycle.mark_running()

    assert lifecycle.begin_stop() is State.STOPPING
    assert lifecycle.mark_stopped() is State.STOPPED
    assert lifecycle.is_terminal is True
    assert lifecycle.is_running is False


def test_failure_records_reason_and_can_still_shut_down() -> None:
    lifecycle = Lifecycle()
    lifecycle.start()

    assert lifecycle.fail("backend unavailable") is State.FAILED
    assert lifecycle.failure_reason == "backend unavailable"

    assert lifecycle.shutdown_cleanly() is State.STOPPED


def test_transition_listener_receives_changes() -> None:
    seen: list[tuple[str, str]] = []
    lifecycle = Lifecycle(on_transition=lambda a, b: seen.append((a.value, b.value)))

    lifecycle.start()
    lifecycle.mark_running()
    lifecycle.begin_stop()
    lifecycle.mark_stopped()

    assert seen == [
        ("new", "starting"),
        ("starting", "running"),
        ("running", "stopping"),
        ("stopping", "stopped"),
    ]


def test_context_manager_starts_and_stops() -> None:
    with Lifecycle() as lifecycle:
        assert lifecycle.is_running is True
    assert lifecycle.state is State.STOPPED


def test_context_manager_fails_on_exception() -> None:
    lifecycle = Lifecycle()
    with pytest.raises(RuntimeError, match="boom"), lifecycle:
        raise RuntimeError("boom")

    assert lifecycle.state is State.STOPPED
    assert lifecycle.failure_reason == "boom"


def test_reset_only_from_terminal_state() -> None:
    lifecycle = Lifecycle()
    with pytest.raises(InvalidTransitionError):
        lifecycle.reset()

    lifecycle.start()
    lifecycle.mark_running()
    lifecycle.shutdown_cleanly()
    assert lifecycle.reset() is State.NEW
