"""Single-instance guard: acquire, refuse, reclaim stale - never kill."""

from __future__ import annotations

import json
import os

import pytest

import robanayt_ai.core.single_instance as single_instance
from robanayt_ai.core.single_instance import (
    InstanceAlreadyRunningError,
    SingleInstanceLock,
    read_lock_owner,
)


def test_acquire_and_release(tmp_path) -> None:
    lock = SingleInstanceLock(tmp_path / "jarvis.lock")

    assert lock.acquire() is True
    assert lock.path.is_file()
    assert lock.acquired is True

    lock.release()

    assert lock.acquired is False
    assert not lock.path.exists()


def test_second_lock_object_is_refused(tmp_path) -> None:
    first = SingleInstanceLock(tmp_path / "jarvis.lock")
    assert first.acquire() is True

    second = SingleInstanceLock(tmp_path / "jarvis.lock")

    assert second.acquire() is False
    # Safety: the existing lock is untouched and the holder is unaffected.
    assert first.path.is_file()
    assert first.acquired is True
    first.release()


def test_alive_owner_is_refused_and_its_file_is_left_alone(tmp_path, monkeypatch) -> None:
    path = tmp_path / "jarvis.lock"
    path.write_text(json.dumps({"pid": 4242, "created_at": 0.0}), encoding="utf-8")
    monkeypatch.setattr(single_instance, "process_alive", lambda pid: True)

    lock = SingleInstanceLock(path)

    assert lock.acquire() is False
    assert path.is_file()
    assert read_lock_owner(path).pid == 4242


def test_stale_lock_is_reclaimed(tmp_path) -> None:
    path = tmp_path / "jarvis.lock"
    path.write_text(json.dumps({"pid": 999983, "created_at": 0.0}), encoding="utf-8")

    lock = SingleInstanceLock(path)

    assert lock.acquire() is True
    assert read_lock_owner(path).pid == os.getpid()
    lock.release()


def test_corrupt_lock_is_reclaimed(tmp_path) -> None:
    path = tmp_path / "jarvis.lock"
    path.write_text("not-json-at-all", encoding="utf-8")

    lock = SingleInstanceLock(path)

    assert lock.acquire() is True
    lock.release()


def test_context_manager_releases(tmp_path) -> None:
    path = tmp_path / "jarvis.lock"
    with SingleInstanceLock(path) as lock:
        assert lock.acquired is True
    assert not path.exists()


def test_release_without_acquire_is_safe(tmp_path) -> None:
    lock = SingleInstanceLock(tmp_path / "jarvis.lock")

    lock.release()

    assert lock.acquired is False


def test_read_owner_of_missing_file(tmp_path) -> None:
    assert read_lock_owner(tmp_path / "missing.lock") is None


def test_raise_if_busy(tmp_path) -> None:
    first = SingleInstanceLock(tmp_path / "jarvis.lock")
    first.acquire()
    second = SingleInstanceLock(tmp_path / "jarvis.lock")

    try:
        with pytest.raises(InstanceAlreadyRunningError):
            second.raise_if_busy()
    finally:
        first.release()
