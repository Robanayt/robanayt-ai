"""The single path from the desktop UI into the JARVIS core.

Rules enforced here:

- Only the documented dialogue API is used: `send`, `approve_pending`,
  `cancel_pending`, `pending`, `describe`. There is no second route, so the UI
  cannot bypass the permission policy.
- Long work runs on a worker thread; results come back through a queue that the
  UI drains on its own thread. The UI thread never waits on a model.
- The reported state is always the real one. An idle process reports `idle`;
  nothing is animated "just in case".
- Error text is sanitised before it reaches the screen.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robanayt_ai.ai.errors import sanitize
from robanayt_ai.orchestration import DialogResult, DialogService
from robanayt_ai.orchestration.models import DialogStatus
from robanayt_ai.orchestration.turn import TurnCancellation
from robanayt_ai.surfaces.protocol import (
    BUSY_STATES,
    STATE_ANSWERING,
    STATE_CANCELLED,
    STATE_CANCELLING,
    STATE_ERROR,
    STATE_IDLE,
    STATE_THINKING,
    STATE_TOOL,
    STATE_WAITING,
    ChatMessage,
    PendingView,
    StateView,
    TurnView,
    pending_view,
    state_view,
    turn_view,
    voice_view,
)
from robanayt_ai.voice.models import VOICE_STATE_LABELS, VoiceState
from robanayt_ai.tools import TOOL_BLOCKED, TOOL_EXECUTED

CONVERSATION_LIMIT = 30
# One page of the sidebar history. The list is loaded in pages, never whole.
HISTORY_PAGE_SIZE = 20


@dataclass(frozen=True)
class BridgeEvent:
    """One result pushed from a worker thread to the UI thread."""

    kind: str  # "turn" | "error" | "cancelled" | "reset" | "voice"
    turn: TurnView | None = None
    message: str = ""


class DialogBridge:
    """Owns the transcript and the current state; calls the dialogue layer."""

    def __init__(
        self,
        dialog: DialogService,
        *,
        ai: Any | None = None,
        tools: Any | None = None,
        memory: Any | None = None,
        bus: Any | None = None,
        mode: Any | None = None,
        voice: Any | None = None,
        speak_responses: bool = False,
        clock: Callable[[], float] = time.time,
        background: bool = True,
    ) -> None:
        self._dialog = dialog
        self._ai = ai
        self._tools = tools
        self._memory = memory
        self._mode = mode
        self._voice = voice
        self._speak_responses = bool(speak_responses)
        self._clock = clock
        self._background = background
        self._events: queue.Queue[BridgeEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._state = STATE_IDLE
        self._messages: list[ChatMessage] = []
        self._conversation_id: str | None = None
        self._last_error: str | None = None
        self._cancellation: TurnCancellation | None = None
        self._voice_report_cache: dict[str, Any] | None = None
        self._closed = False
        self._threads: list[threading.Thread] = []
        self._unsubscribes: list[Callable[[], None]] = []
        if bus is not None:
            for event_type in (TOOL_EXECUTED, TOOL_BLOCKED):
                try:
                    self._unsubscribes.append(bus.subscribe(event_type, self._on_tool_event))
                except Exception:  # noqa: BLE001 - a bus without the event is fine
                    continue

    # -- commands (called on the UI thread) ---------------------------------

    def send(self, text: str, *, conversation_id: str | None = None) -> bool:
        """Start one turn. Empty input is refused, not sent."""
        message = (text or "").strip()
        if not message or self._closed:
            return False
        self._append(ChatMessage(role="user", text=message))
        self._set_state(STATE_THINKING)
        target = conversation_id or self._conversation_id
        cancellation = TurnCancellation()
        self._cancellation = cancellation
        return self._start(
            lambda: self._dialog.send(
                message,
                conversation_id=target,
                cancellation=cancellation,
                observer=self,
            )
        )

    def approve(self, pending_id: str) -> bool:
        if not pending_id or self._closed:
            return False
        self._set_state(STATE_TOOL)
        cancellation = self._cancellation or TurnCancellation()
        self._cancellation = cancellation
        return self._start(
            lambda: self._dialog.approve_pending(
                pending_id, cancellation=cancellation, observer=self
            )
        )

    def cancel_turn(self) -> bool:
        """Ask the running turn to stop at its next safe point.

        Returns `False` when nothing is running: cancelling nothing is not a
        success. A tool that is already executing is *not* interrupted - the
        resulting turn still lists what ran, so the UI never claims that a
        started action was undone.
        """
        if self._closed or self._cancellation is None or not self.is_busy:
            return False
        self._cancellation.cancel()
        self._set_state(STATE_CANCELLING)
        return True

    # -- voice (explicit actions only) --------------------------------------

    def start_voice(self) -> bool:
        """Listen once; the transcript then follows the ordinary text path.

        Nothing records by itself. A recognised phrase is sent exactly like
        typed text, so permissions, approvals and refusals stay identical.
        """
        if self._closed or self._voice is None or self.is_busy:
            return False
        # The dialogue may be idle while the microphone is already open: a
        # second capture would fight the first one for the same device.
        if getattr(self._voice, "busy", False):
            return False
        return self._spawn(self._voice_worker)

    def cancel_voice(self) -> bool:
        """Ask voice to stop. Cooperative: between steps, not mid-hardware."""
        if self._voice is None:
            return False
        try:
            return bool(self._voice.cancel())
        except Exception:  # noqa: BLE001 - never break the UI
            return False

    def speak(self, text: str) -> bool:
        """Speak one piece of text. Empty text is refused, not played."""
        if self._voice is None:
            return False
        try:
            return bool(self._voice.speak(text or ""))
        except Exception:  # noqa: BLE001 - reported through the event queue
            return False

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._state in BUSY_STATES

    # -- progress observation (called from the turn's own thread) -----------

    def on_model_start(self) -> None:
        if self._cancel_requested():
            return
        self._set_state(STATE_THINKING)

    def on_model_done(self, *, has_tool_call: bool) -> None:
        if self._cancel_requested():
            return
        self._set_state(STATE_THINKING if has_tool_call else STATE_ANSWERING)

    def cancel(self, pending_id: str) -> bool:
        if self._closed:
            return False
        dropped = self._dialog.cancel_pending(pending_id)
        self._events.put(BridgeEvent(kind="cancelled", message=pending_id if dropped else ""))
        if dropped and not self._dialog.pending():
            self._set_state(STATE_IDLE)
        return dropped

    def new_conversation(self) -> None:
        self._conversation_id = None
        self._cancellation = None
        with self._lock:
            self._messages = []
        self._last_error = None
        self._set_state(STATE_IDLE)
        self._events.put(BridgeEvent(kind="reset"))

    def conversations(
        self, limit: int = HISTORY_PAGE_SIZE, offset: int = 0
    ) -> tuple[tuple[str, str], ...]:
        """One page of the sidebar history. Metadata only - no message text.

        The list is always loaded in pages: a long history never becomes an
        unbounded query or an unbounded UI list.
        """
        if self._memory is None:
            return ()
        try:
            items = self._memory.conversations.list_conversations(
                limit=max(0, int(limit)), offset=max(0, int(offset))
            )
        except Exception:  # noqa: BLE001 - memory may be disabled or broken
            return ()
        return tuple((item.id, item.title or "") for item in items)

    def export_conversation(
        self,
        conversation_id: str | None = None,
        path: str | Path | None = None,
        *,
        overwrite: bool = True,
    ) -> tuple[bool, str]:
        """Export one conversation through the memory layer.

        Returns `(ok, message)`. Storage is not duplicated: the export uses
        `MemoryService.export_conversation_to_file`, which validates the output
        path. Failures are returned, never raised, so the UI can report them.
        """
        target = conversation_id or self._conversation_id
        if not target:
            return (False, "нет диалога для экспорта")
        if self._memory is None:
            return (False, "память отключена: экспорт недоступен")
        try:
            destination = (
                Path(path)
                if path
                else Path(self._memory.config.export_directory) / f"{target}.json"
            )
            written = self._memory.export_conversation_to_file(
                target, destination, overwrite=overwrite
            )
        except Exception as exc:  # noqa: BLE001 - reported to the user, not raised
            return (False, sanitize(str(exc)))
        return (True, str(written))

    def open_conversation(self, conversation_id: str) -> bool:
        """Load an existing conversation into the transcript."""
        if self._memory is None or not conversation_id:
            return False
        try:
            self._memory.conversations.require_conversation(conversation_id)
            stored = self._memory.conversations.get_messages(conversation_id)
        except Exception:  # noqa: BLE001 - unknown id or unavailable storage
            return False
        with self._lock:
            self._messages = [
                ChatMessage(role=item.role, text=item.content or "") for item in stored
            ]
        self._conversation_id = conversation_id
        self._set_state(STATE_IDLE)
        self._events.put(BridgeEvent(kind="reset"))
        return True

    # -- state (read on the UI thread) --------------------------------------

    def poll(self) -> tuple[BridgeEvent, ...]:
        """Drain everything the workers produced since the last call."""
        events: list[BridgeEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return tuple(events)

    def snapshot(self) -> StateView:
        with self._lock:
            messages = tuple(self._messages)
            state = self._state
        pending: list[PendingView] = []
        try:
            pending = [pending_view(item) for item in self._dialog.pending()]
        except Exception:  # noqa: BLE001 - never break the UI on introspection
            pending = []
        limits: dict[str, Any] = {}
        try:
            limits = dict(self._dialog.describe())
        except Exception:  # noqa: BLE001
            limits = {}
        return state_view(
            state=state,
            conversation_id=self._conversation_id,
            messages=messages,
            pending=tuple(pending),
            providers=self._provider_lines(),
            mode=self._mode_name(),
            tools=self._tool_lines(),
            limits=limits,
            last_error=self._last_error,
            voice=self._voice_view(),
        )

    def info(self) -> dict[str, Any]:
        """Diagnostics for the settings window: only what is really wired."""
        data: dict[str, Any] = {"mode": self._mode_name()}
        if self._ai is not None:
            try:
                data["ai"] = dict(self._ai.describe())
            except Exception:  # noqa: BLE001
                data["ai"] = {}
        if self._tools is not None:
            try:
                data["tools"] = dict(self._tools.describe())
            except Exception:  # noqa: BLE001
                data["tools"] = {}
        if self._memory is not None:
            try:
                data["memory"] = {
                    "enabled": bool(self._memory.enabled),
                    "database": str(self._memory.database_path),
                }
            except Exception:  # noqa: BLE001
                data["memory"] = {}
        try:
            data["dialog"] = dict(self._dialog.describe())
        except Exception:  # noqa: BLE001
            data["dialog"] = {}
        data["providers"] = self._provider_lines()
        data["tools_list"] = self._tool_lines()
        data["voice"] = self._voice_report()
        return data

    def close(self) -> None:
        self._closed = True
        if self._voice is not None:
            try:
                self._voice.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass
        for unsubscribe in self._unsubscribes:
            try:
                unsubscribe()
            except Exception:  # noqa: BLE001
                continue
        self._unsubscribes.clear()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._threads.clear()

    # -- internals ----------------------------------------------------------

    def _voice_report(self) -> dict[str, Any]:
        """Voice diagnostics: what exists, what is available, and why not."""
        if self._voice is None:
            return {"enabled": False, "reason": "голосовая подсистема не подключена"}
        try:
            cached = self._voice_report_cache
        except Exception:  # pragma: no cover - defensive
            cached = None
        cached = self._voice_report_cache
        if cached is not None:
            report = dict(cached)
        else:
            try:
                report = dict(self._voice.explain(probe=False))
            except Exception:  # noqa: BLE001 - diagnostics never break the UI
                return {"enabled": False, "reason": "отчёт недоступен"}
        # Providers and availability come from the (cached) probe; the state
        # itself costs nothing to read, and freezing it would let the UI claim
        # "microphone idle" while the microphone is open.
        report["state"] = self._live_voice_state(report)
        return report

    def _live_voice_state(self, report: dict[str, Any]) -> str:
        """The probe is cached; the current state must not be."""
        if not report.get("enabled", True):
            return VoiceState.UNAVAILABLE.value
        live = getattr(self._voice, "state", None)
        if isinstance(live, str) and live:
            return live
        return str(report.get("state") or VoiceState.IDLE.value)

    def probe_voice(self) -> bool:
        """Refresh availability in a worker thread - never on the UI thread."""
        if self._voice is None or self._closed:
            return False

        def worker() -> None:
            try:
                report = dict(self._voice.explain(probe=True))
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                report = {"enabled": False, "reason": sanitize(str(exc))}
            self._voice_report_cache = report
            self._events.put(BridgeEvent(kind="voice", message="probe"))

        return self._spawn(worker)

    def _voice_view(self) -> Any | None:
        if self._voice is None:
            return None
        report = self._voice_report()
        try:
            state = str(report.get("state") or VoiceState.IDLE.value)
        except Exception:  # pragma: no cover - defensive
            state = VoiceState.IDLE.value
        stt = report.get("stt") or {}
        tts = report.get("tts") or {}
        stt_name = stt.get("selected")
        tts_name = tts.get("selected")
        available = bool(stt_name) or bool(tts_name)
        if available:
            detail = f"STT: {stt_name or 'нет'} · TTS: {tts_name or 'нет'}"
            if not stt_name:
                # A missing recogniser disables listening only, not speaking:
                # naming the side is better than a bare "unavailable".
                detail = "распознавание недоступно · " + detail
        else:
            detail = str(
                tts.get("reason")
                or stt.get("reason")
                or report.get("reason")
                or "голос недоступен"
            )
        return voice_view(
            state=state,
            label=VOICE_STATE_LABELS.get(state, state),
            available=available,
            stt_provider=stt_name,
            tts_provider=tts_name,
            detail=detail,
        )

    def _voice_worker(self) -> None:
        """Background voice capture, then the ordinary text pipeline."""
        try:
            turn = self._voice.transcribe()
        except Exception as exc:  # noqa: BLE001 - surfaced, never raised
            self._fail(sanitize(str(exc)))
            return
        if turn.error or turn.transcript_empty or not turn.transcript:
            detail = turn.error or "речь не распознана"
            # Nothing was understood: nothing is sent, and nothing is stored.
            self._events.put(BridgeEvent(kind="voice", message=detail))
            return

        message = turn.transcript.strip()
        self._append(ChatMessage(role="user", text=message))
        self._set_state(STATE_THINKING)
        cancellation = TurnCancellation()
        self._cancellation = cancellation
        try:
            result = self._dialog.send(
                message,
                conversation_id=self._conversation_id,
                cancellation=cancellation,
                observer=self,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced, never raised
            self._fail(sanitize(str(exc)))
            return
        self._apply(result)
        if (
            self._speak_responses
            and result.status is DialogStatus.SUCCESS
            and result.answer
        ):
            try:
                self._voice.speak(result.answer)
            except Exception as exc:  # noqa: BLE001 - speaking failure is not fatal
                self._events.put(
                    BridgeEvent(kind="voice", message=f"озвучивание не удалось: {sanitize(str(exc))}")
                )

    def _spawn(self, worker: Callable[[], None]) -> bool:
        if self._background:
            thread = threading.Thread(target=worker, daemon=True)
            with self._lock:
                self._threads.append(thread)
            thread.start()
        else:
            worker()
        return True

    def _start(self, work: Callable[[], DialogResult]) -> bool:
        def worker() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - surface, never crash
                self._fail(sanitize(str(exc)))
                return
            self._apply(result)

        if self._background:
            thread = threading.Thread(target=worker, daemon=True)
            with self._lock:
                self._threads.append(thread)
            thread.start()
        else:
            worker()
        return True

    def _apply(self, result: DialogResult) -> None:
        view = turn_view(result)
        if result.conversation_id:
            self._conversation_id = result.conversation_id

        if view.pending is not None:
            self._set_state(STATE_WAITING)
            self._append(
                ChatMessage(
                    role="system",
                    text=f"Требуется подтверждение: {view.pending.tool}",
                    meta=view.pending.pending_id,
                )
            )
        elif result.status is DialogStatus.CANCELLED:
            executed = len(result.tool_calls)
            note = "Ход отменён."
            if executed:
                note += (
                    f" Уже выполнено действий: {executed}. "
                    "Запущенное действие не отменяется — это результат, а не обещание."
                )
            self._append(ChatMessage(role="system", text=note, meta="Отмена"))
            self._last_error = None
            self._set_state(STATE_CANCELLED)
        elif result.status is DialogStatus.SUCCESS:
            self._append(
                ChatMessage(role="assistant", text=result.answer or "(пустой ответ)")
            )
            self._last_error = None
            self._set_state(STATE_IDLE)
        else:
            failure = sanitize(result.error or str(getattr(result.status, "value", result.status)))
            self._last_error = failure
            self._append(
                ChatMessage(role="system", text=failure, meta=result.error_type or "")
            )
            self._set_state(STATE_ERROR)
        self._cancellation = None
        self._events.put(BridgeEvent(kind="turn", turn=view))

    def _fail(self, message: str) -> None:
        self._last_error = message
        self._cancellation = None
        self._append(ChatMessage(role="system", text=message, meta="ошибка"))
        self._set_state(STATE_ERROR)
        self._events.put(BridgeEvent(kind="error", message=message))

    def _cancel_requested(self) -> bool:
        cancellation = self._cancellation
        return cancellation is not None and cancellation.requested

    def _append(self, message: ChatMessage) -> None:
        with self._lock:
            self._messages.append(message)

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def _on_tool_event(self, _event: Any) -> None:
        if self._cancel_requested():
            return
        self._set_state(STATE_TOOL)

    def _provider_lines(self) -> tuple[str, ...]:
        if self._ai is None:
            return ()
        try:
            statuses = self._ai.status()
        except Exception:  # noqa: BLE001
            return ()
        lines: list[str] = []
        for status in statuses:
            state = "доступен" if status.available else "недоступен"
            lines.append(f"{status.provider}: {state}")
        return tuple(lines)

    def _tool_lines(self) -> tuple[tuple[str, str], ...]:
        if self._tools is None:
            return ()
        try:
            registry = self._tools.registry
            return tuple(
                (spec.name, str(getattr(spec.permission, "value", spec.permission)))
                for spec in registry.specs()
            )
        except Exception:  # noqa: BLE001
            return ()

    def _mode_name(self) -> str:
        value = getattr(self._mode, "value", self._mode)
        return str(value) if value else "normal"


def format_messages(messages: Sequence[ChatMessage]) -> tuple[str, ...]:
    """Rendering helper: `"role: text"` lines, used by tests and headless UI."""
    return tuple(f"{message.role}: {message.text}" for message in messages)
