"""Single-instance guard.

A lock file holds the owning PID. Acquiring it is atomic (`O_EXCL`), so two
processes cannot both win. If an existing lock is held by a live process the
caller simply gets `False` - this module never kills, signals or otherwise
touches another process. A lock left by a dead process is stale and may be
reclaimed; a lock whose state cannot be determined is left alone (safe
default: report "busy").

Windows-friendly: no named mutexes, no POSIX-only calls, no third-party
packages. PID liveness on Windows uses a query-only `OpenProcess`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class InstanceAlreadyRunningError(RuntimeError):
    """Raised by callers that prefer an exception over a boolean."""


@dataclass(frozen=True)
class LockOwner:
    """Whoever wrote the lock file."""

    pid: int
    created_at: float
    app: str = "robanayt_ai"
    pid_alive: bool = False


def process_alive(pid: int) -> bool:
    """True if the PID exists; unknown is treated as alive (safe default)."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _windows_process_alive(pid: int) -> bool:
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except Exception:  # pragma: no cover - non-Windows / no ctypes
        return True
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


class SingleInstanceLock:
    """Context manager around the instance lock file."""

    def __init__(self, path: str | Path, *, pid: int | None = None) -> None:
        self.path = Path(path)
        self.pid = pid if pid is not None else os.getpid()
        self._handle: int | None = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """Take the lock. Returns False when another instance holds it."""
        if self._handle is not None:
            return True

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("cannot create lock directory %s: %s", self.path.parent, exc)
            return False

        for attempt in (0, 1):
            try:
                fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                if attempt == 1:
                    return False
                if not self._clear_stale_lock():
                    return False
                continue
            except OSError as exc:
                logger.error("cannot create lock file %s: %s", self.path, exc)
                return False

            try:
                os.write(fd, self._encode_owner())
            except OSError as exc:  # pragma: no cover - unusual I/O failure
                logger.error("cannot write lock file %s: %s", self.path, exc)
                os.close(fd)
                self._remove_lock_file()
                return False
            self._handle = fd
            logger.debug("instance lock acquired: %s (pid=%s)", self.path, self.pid)
            return True
        return False

    def release(self) -> None:
        """Release the lock if this object owns it."""
        if self._handle is None:
            return
        try:
            os.close(self._handle)
        except OSError as exc:  # pragma: no cover - defensive
            logger.debug("closing lock handle failed: %s", exc)
        self._handle = None
        self._remove_lock_file()
        logger.debug("instance lock released: %s", self.path)

    def read_owner(self) -> LockOwner | None:
        """Parse the lock file, or None when absent/unreadable."""
        return read_lock_owner(self.path)

    def raise_if_busy(self) -> None:
        """Convenience for callers that want an exception instead of False."""
        if not self.acquire():
            raise InstanceAlreadyRunningError(
                f"another Robanayt AI instance is already running (lock: {self.path})"
            )

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    def _encode_owner(self) -> bytes:
        owner = {
            "pid": self.pid,
            "created_at": time.time(),
            "app": "robanayt_ai",
        }
        return json.dumps(owner).encode("utf-8")

    def _clear_stale_lock(self) -> bool:
        """Remove a lock only when it is provably stale. True if removable."""
        owner = read_lock_owner(self.path)
        if owner is None:
            logger.warning(
                "lock file %s is unreadable or empty; treating as stale", self.path
            )
            self._remove_lock_file()
            return True
        if owner.pid == self.pid:
            logger.error(
                "lock file %s is already owned by this process (pid=%s)",
                self.path,
                self.pid,
            )
            return False
        if process_alive(owner.pid):
            logger.error(
                "another instance is already running (pid=%s, lock=%s)",
                owner.pid,
                self.path,
            )
            return False
        logger.warning(
            "removing stale lock %s (dead pid=%s)", self.path, owner.pid
        )
        self._remove_lock_file()
        return True

    def _remove_lock_file(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - defensive
            logger.debug("cannot remove lock file %s: %s", self.path, exc)


def read_lock_owner(path: str | Path) -> LockOwner | None:
    """Read and parse a lock file; None when missing or corrupt."""
    lock_path = Path(path)
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("lock file %s is not valid JSON; treating as stale", lock_path)
        return None
    if not isinstance(data, dict) or "pid" not in data:
        return None
    try:
        pid = int(data["pid"])
    except (TypeError, ValueError):
        return None
    created_raw = data.get("created_at")
    try:
        created_at = float(created_raw) if created_raw is not None else 0.0
    except (TypeError, ValueError):
        created_at = 0.0
    return LockOwner(
        pid=pid,
        created_at=created_at,
        app=str(data.get("app", "robanayt_ai")),
        pid_alive=process_alive(pid),
    )
