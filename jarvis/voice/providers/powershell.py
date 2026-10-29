"""Host for running Windows PowerShell helpers.

Why this exists: Windows ships speech synthesis (System.Speech / SAPI) as part
of the OS, so text-to-speech needs **no new Python dependency**. The only way
to reach it from the standard library is to ask Windows itself.

Safety rules enforced here:

- The script always travels as `-EncodedCommand` (base64 of UTF-16LE), so no
  quoting or shell metacharacter of any payload can be interpreted.
- Every payload (for example the text to speak) is additionally passed as
  base64 and decoded inside the script - never interpolated as code.
- No shell is used, stdin is fed as bytes, and every run has a timeout.
- Cancellation terminates *our own child process*; it is honest but abrupt:
  audio already played cannot be taken back, so capabilities say so.
"""

from __future__ import annotations

import base64
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_EXECUTABLES: tuple[str, ...] = ("powershell", "pwsh")
POLL_SECONDS = 0.2


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one helper run - no secrets, no audio, only text."""

    ok: bool
    code: int
    output: str = ""
    error: str = ""
    timed_out: bool = False
    cancelled: bool = False

    def lines(self) -> tuple[str, ...]:
        return tuple(line for line in self.output.splitlines() if line.strip())


def encode_script(script: str) -> str:
    """`-EncodedCommand` payload: base64 of UTF-16LE."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def encode_text(text: str) -> str:
    """Payload helper: base64 of UTF-8, decoded back inside the script."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


PROBE_DEVICES = """\
$ErrorActionPreference='Stop'
foreach ($device in (Get-CimInstance Win32_SoundDevice)) {
  'DEVICE|' + $device.Name
}
"""


def probe_audio_devices(
    runner: Callable[..., CommandResult] | None = None,
    *,
    timeout: float = 10.0,
) -> tuple[str, ...]:
    """Read-only WMI listing of sound devices. Never records, never plays.

    Returns an empty tuple when Windows refuses the query - an unknown device
    list is reported as "unknown", never as "no microphone".
    """
    host = runner or SubprocessPowerShellRunner()
    try:
        result = host(PROBE_DEVICES, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - diagnostics must degrade quietly
        logger.debug("audio device probe failed: %s", exc)
        return ()
    return tuple(
        line.split("|", 1)[1]
        for line in result.lines()
        if line.startswith("DEVICE|") and len(line.split("|", 1)) > 1
    )


def discover_executable(names: Iterable[str] = DEFAULT_EXECUTABLES) -> str | None:
    """Return the first available host, or `None` when there is none."""
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


class SubprocessPowerShellRunner:
    """Runs a script through PowerShell with timeout and cancellation."""

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or discover_executable() or DEFAULT_EXECUTABLES[0]

    def __call__(
        self,
        script: str,
        *,
        timeout: float,
        stdin: str = "",
        cancel: Callable[[], bool] | None = None,
    ) -> CommandResult:
        command = [
            self.executable,
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encode_script(script),
        ]
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:  # pragma: no cover - host missing/broken
            logger.warning("powershell host unavailable: %s", exc)
            return CommandResult(ok=False, code=-1, error=f"не удалось запустить хост: {exc}")

        payload = stdin.encode("utf-8") if stdin else b""
        try:
            if payload:
                assert process.stdin is not None
                process.stdin.write(payload)
                process.stdin.close()
            return self._wait(process, timeout=timeout, cancel=cancel)
        except Exception as exc:  # noqa: BLE001 - never crash the caller
            logger.error("powershell run failed: %s", exc)
            self._terminate(process)
            return CommandResult(ok=False, code=-1, error=str(exc))

    def _wait(
        self,
        process: subprocess.Popen[bytes],
        *,
        timeout: float,
        cancel: Callable[[], bool] | None,
    ) -> CommandResult:
        deadline_remaining = max(0.1, float(timeout))
        while True:
            try:
                stdout, stderr = process.communicate(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                deadline_remaining -= POLL_SECONDS
                if cancel is not None and cancel():
                    self._terminate(process)
                    process.communicate()
                    return CommandResult(
                        ok=False, code=-1, output="", error="отменено", cancelled=True
                    )
                if deadline_remaining <= 0:
                    self._terminate(process)
                    process.communicate()
                    return CommandResult(
                        ok=False, code=-1, output="", error="время истекло", timed_out=True
                    )
                continue
            text_out = _decode(stdout)
            text_err = _decode(stderr)
            return CommandResult(
                ok=process.returncode == 0,
                code=int(process.returncode or 0),
                output=text_out,
                error=text_err,
            )

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            process.terminate()
        except OSError:  # pragma: no cover - already gone
            return


def _decode(raw: bytes | None) -> str:
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").strip()
