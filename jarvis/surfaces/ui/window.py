"""Main window: sidebar, core, transcript, input and inspector.

Layout follows the product rules: history and settings on the left, the core
and the dialogue in the middle, live state and approvals on the right. Every
element shows real data from `DialogBridge` - no decorative progress.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

from jarvis.surfaces.bridge import HISTORY_PAGE_SIZE, DialogBridge
from jarvis.surfaces.protocol import (
    STATE_ANSWERING,
    STATE_CANCELLED,
    STATE_ERROR,
    STATE_IDLE,
    STATE_THINKING,
    STATE_TOOL,
    PendingView,
    StateView,
)
from jarvis.surfaces.theme import DEFAULT_THEME, REDUCED_MOTION_THEME, Theme
from jarvis.surfaces.ui.approval import ApprovalDialog
from jarvis.surfaces.ui.core_widget import CoreWidget
from jarvis.surfaces.ui.settings import DEFERRED, open_settings
from jarvis.surfaces.ui.widgets import (
    apply_theme,
    button,
    label,
    panel,
    scrolled_text,
)

TITLE = "Robanayt AI"
WELCOME = (
    "Robanayt AI готов к работе.\n"
    "Задайте задачу обычным языком. Если действие требует разрешения, "
    "оно появится в правой панели и будет ждать вашего решения.\n"
)


class HolographicApp:
    """The desktop window. Owns no core object - it only holds the bridge."""

    def __init__(
        self,
        bridge: DialogBridge,
        *,
        theme: Theme | None = None,
        hidden: bool = False,
    ) -> None:
        self._bridge = bridge
        self._theme = theme or DEFAULT_THEME
        self._hidden = hidden
        self._closed = False
        self._rendered = -1
        self._pending_ids: tuple[str, ...] = ()
        self._history_ids: list[str] = []
        self._history_offset = 0
        self._page_size = HISTORY_PAGE_SIZE
        self._jobs: queue.Queue[tuple[bool, str]] = queue.Queue()
        self._terminal_at: float | None = None

        self._root = tk.Tk()
        self._root.title(TITLE)
        self._root.geometry("1280x840")
        self._root.minsize(1020, 660)
        if hidden:  # smoke tests build the window without showing it
            self._root.withdraw()
        apply_theme(self._root, self._theme)
        self._build()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._core.start()
        # Availability is probed once, in a worker thread: the first frame must
        # not wait for Windows to answer.
        self._bridge.probe_voice()
        self._poll()

    # -- lifecycle ----------------------------------------------------------

    @property
    def root(self) -> tk.Misc:
        """The Tk root - exposed so smoke tests can drive the window."""
        return self._root

    def run(self) -> None:
        self._root.mainloop()

    def close(self) -> None:
        self._on_close()

    def _on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._core.stop()
        self._bridge.close()
        try:
            self._root.destroy()
        except tk.TclError:  # pragma: no cover - already destroyed
            pass

    # -- construction -------------------------------------------------------

    def _build(self) -> None:
        metrics = self._theme.metrics
        self._root.columnconfigure(0, minsize=metrics.sidebar, weight=0)
        self._root.columnconfigure(1, weight=1)
        self._root.columnconfigure(2, minsize=metrics.inspector, weight=0)
        self._root.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_center()
        self._build_inspector()

    def _build_sidebar(self) -> None:
        palette = self._theme.palette
        side = panel(self._root, self._theme)
        side.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)

        label(side, "ROBANAYT AI", style_name="Title.TLabel").pack(anchor="w", pady=(4, 12))
        button(side, "Новый диалог", self._new_conversation).pack(fill="x", pady=(0, 10))

        label(side, "История", style_name="Section.TLabel").pack(anchor="w", pady=(4, 4))
        self._history = tk.Listbox(
            side,
            height=14,
            background=palette.surface,
            foreground=palette.text,
            selectbackground=palette.accent_dim,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=(self._theme.type.family, self._theme.type.small),
        )
        self._history.pack(fill="both", expand=True)
        self._history.bind("<<ListboxSelect>>", self._on_select_conversation)
        self._more_button = button(side, "Показать ещё", self._load_more_history)
        self._more_button.pack(fill="x", pady=(4, 0))
        button(side, "Экспорт диалога", self._export_conversation).pack(fill="x", pady=(4, 0))

        label(side, "Функции", style_name="Section.TLabel").pack(anchor="w", pady=(14, 4))
        # "Голос" is no longer a deferred stub: it has a real state below.
        for name in ("Зрение", "Автоматизация Windows", "Веб-исследования"):
            label(
                side,
                f"• {name} — {DEFERRED}",
                style_name="Panel.Dim.TLabel",
                wraplength=self._theme.metrics.sidebar - 44,
            ).pack(anchor="w", pady=1)

        label(side, "Голос", style_name="Section.TLabel").pack(anchor="w", pady=(14, 4))
        self._voice_var = tk.StringVar(value="состояние: не проверено")
        label(side, textvariable=self._voice_var).pack(anchor="w")
        self._voice_detail_var = tk.StringVar(value="")
        label(
            side,
            textvariable=self._voice_detail_var,
            style_name="Panel.Dim.TLabel",
            wraplength=self._theme.metrics.sidebar - 44,
        ).pack(anchor="w", pady=(0, 4))
        voice_row = ttk.Frame(side, style="Panel.TFrame")
        voice_row.pack(fill="x")
        self._voice_listen_button = button(voice_row, "Слушать", self._start_voice)
        self._voice_listen_button.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._voice_stop_button = button(
            voice_row, "Остановить", self._cancel_voice, danger=True
        )
        self._voice_stop_button.pack(side="left")
        self._voice_stop_button.configure(state="disabled")
        self._voice_speak_button = button(side, "Озвучить ответ", self._speak_last_answer)
        self._voice_speak_button.pack(fill="x", pady=(4, 0))

        button(side, "Настройки", self._open_settings).pack(fill="x", side="bottom", pady=(12, 0))
        self._refresh_history(reset=True)

    def _build_center(self) -> None:
        palette = self._theme.palette
        center = ttk.Frame(self._root)
        center.grid(row=0, column=1, sticky="nsew", padx=6, pady=10)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        head = panel(center, self._theme)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._core = CoreWidget(head, self._theme)
        self._core.pack(pady=(4, 2))
        self._status_var = tk.StringVar(value="ожидание")
        label(head, textvariable=self._status_var).pack()
        self._provider_var = tk.StringVar(value="")
        label(head, textvariable=self._provider_var, style_name="Panel.Dim.TLabel").pack(pady=(0, 4))

        _frame, self._transcript = scrolled_text(center, self._theme, height=18)
        _frame.grid(row=1, column=0, sticky="nsew")
        for tag, colour in (
            ("user", palette.accent),
            ("assistant", palette.text),
            ("system", palette.text_dim),
            ("error", palette.error),
        ):
            self._transcript.tag_configure(tag, foreground=colour)

        entry = panel(center, self._theme)
        entry.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        entry.columnconfigure(0, weight=1)
        self._input = tk.Text(
            entry,
            height=3,
            wrap="word",
            background=palette.surface,
            foreground=palette.text,
            insertbackground=palette.accent,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=palette.border,
            font=(self._theme.type.family, self._theme.type.body),
            padx=10,
            pady=8,
        )
        self._input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._input.bind("<Return>", self._on_send)
        button(entry, "Отправить", self._send, accent=True).grid(row=0, column=1)
        self._cancel_button = button(entry, "Отмена хода", self._cancel_turn, danger=True)
        self._cancel_button.grid(row=0, column=2, padx=(8, 0))
        self._cancel_button.configure(state="disabled")
        self._hint_var = tk.StringVar(value="Enter — отправить, Shift+Enter — новая строка")
        label(entry, textvariable=self._hint_var, style_name="Panel.Dim.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        self._transcript.configure(state="normal")
        self._transcript.insert("1.0", WELCOME, ("system",))
        self._transcript.configure(state="disabled")

    def _build_inspector(self) -> None:
        right = panel(self._root, self._theme)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 10), pady=10)

        label(right, "Состояние", style_name="Section.TLabel").pack(anchor="w")
        self._mode_var = tk.StringVar(value="—")
        label(right, textvariable=self._mode_var, style_name="Panel.Dim.TLabel").pack(anchor="w")
        self._conversation_var = tk.StringVar(value="—")
        label(right, textvariable=self._conversation_var, style_name="Panel.Dim.TLabel").pack(
            anchor="w"
        )
        self._limits_var = tk.StringVar(value="")
        label(
            right,
            textvariable=self._limits_var,
            style_name="Panel.Dim.TLabel",
            wraplength=self._theme.metrics.inspector - 40,
        ).pack(anchor="w", pady=(0, 10))

        label(right, "Ожидают подтверждения", style_name="Section.TLabel").pack(anchor="w")
        self._pending_frame = ttk.Frame(right, style="Panel.TFrame")
        self._pending_frame.pack(fill="x", pady=(4, 10))
        self._pending_empty = label(right, "нет", style_name="Panel.Dim.TLabel")
        self._pending_empty.pack(anchor="w", pady=(0, 10))

        label(right, "Инструменты", style_name="Section.TLabel").pack(anchor="w")
        self._tools_box = tk.Listbox(
            right,
            height=8,
            background=self._theme.palette.surface,
            foreground=self._theme.palette.text,
            selectbackground=self._theme.palette.accent_dim,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=(self._theme.type.family, self._theme.type.small),
        )
        self._tools_box.pack(fill="x", pady=(4, 10))

        label(right, "Диагностика", style_name="Section.TLabel").pack(anchor="w")
        self._error_var = tk.StringVar(value="—")
        label(
            right,
            textvariable=self._error_var,
            style_name="Panel.Dim.TLabel",
            wraplength=self._theme.metrics.inspector - 40,
        ).pack(anchor="w", pady=(4, 0))

    # -- refresh loop -------------------------------------------------------

    def _poll(self) -> None:
        if self._closed:
            return
        for event in self._bridge.poll():
            if event.kind == "reset":
                self._rendered = -1
                self._refresh_history(reset=True)
            elif event.kind == "turn" and event.turn is not None:
                if event.turn.status in {"cancelled", "error"}:
                    self._hint_var.set(self._outcome_hint(event.turn))
            elif event.kind == "voice" and event.message not in {"", "probe"}:
                self._hint_var.set(f"Голос: {event.message}")
        while True:  # background jobs (export) report here, on the UI thread
            try:
                ok, message = self._jobs.get_nowait()
            except queue.Empty:
                break
            self._hint_var.set(f"Экспорт: {message}" if ok else f"Ошибка экспорта: {message}")
        view = self._bridge.snapshot()
        self._core.set_state(self._core_state(view.state))
        self._cancel_button.configure(
            state="normal"
            if view.state in {STATE_THINKING, STATE_ANSWERING, STATE_TOOL}
            else "disabled"
        )
        self._status_var.set(view.state_label)
        self._provider_var.set(" · ".join(view.providers) or "провайдер не настроен")
        self._mode_var.set(f"режим: {view.mode}")
        self._conversation_var.set(f"диалог: {view.conversation_id or 'новый'}")
        self._limits_var.set(self._limits_text(view))
        self._error_var.set(view.last_error or "ошибок нет")
        self._render_voice(view)
        self._render_transcript(view)
        self._render_pending(view.pending)
        self._render_tools(view)
        self._root.after(self._theme.motion.poll_ms, self._poll)

    TERMINAL_SETTLE_SECONDS = 2.0

    def _core_state(self, state: str) -> str:
        """Terminal states animate briefly, then the core rests at `idle`.

        The status line keeps the real outcome; only the animation settles, so
        a finished error or cancellation does not look like work in progress.
        """
        if state not in {STATE_ERROR, STATE_CANCELLED}:
            self._terminal_at = None
            return state
        now = time.monotonic()
        if self._terminal_at is None:
            self._terminal_at = now
        if now - self._terminal_at < self.TERMINAL_SETTLE_SECONDS:
            return state
        return STATE_IDLE

    @staticmethod
    def _outcome_hint(turn: Any) -> str:
        """One unambiguous line after a failed or cancelled turn."""
        if turn.status == "cancelled":
            return "Ход отменён. Выполненные действия перечислены в транскрипте."
        detail = turn.error or turn.status
        return f"Ход завершился ошибкой ({turn.error_type or 'ошибка'}): {detail}"

    def _load_more_history(self) -> None:
        self._refresh_history(reset=False)

    def _export_conversation(self) -> None:
        conversation_id = self._bridge.snapshot().conversation_id
        target = filedialog.asksaveasfilename(
            parent=self._root,
            title="Экспорт диалога",
            defaultextension=".json",
            initialfile=f"{conversation_id or 'conversation'}.json",
            filetypes=(("JSON", "*.json"),),
        )
        if not target:
            self._hint_var.set("Экспорт отменён.")
            return
        self._hint_var.set("Экспорт: запись файла…")
        threading.Thread(
            target=self._export_worker,
            args=(conversation_id, target),
            daemon=True,
        ).start()

    def _export_worker(self, conversation_id: str | None, target: str) -> None:
        ok, message = self._bridge.export_conversation(conversation_id, target)
        self._jobs.put((ok, message))

    # -- voice --------------------------------------------------------------

    def _render_voice(self, view: StateView) -> None:
        """Show the real voice state, and nothing else."""
        voice = getattr(view, "voice", None)
        if voice is None:
            self._voice_var.set("голос не подключён")
            self._voice_detail_var.set("")
            for widget in (
                self._voice_listen_button,
                self._voice_stop_button,
                self._voice_speak_button,
            ):
                widget.configure(state="disabled")
            return
        self._voice_var.set(f"состояние: {voice.label}")
        detail = voice.detail
        if voice.available and "STT:" not in detail:
            # The bridge already names both sides; never print it twice.
            providers = f"STT: {voice.stt_provider or 'нет'} · TTS: {voice.tts_provider or 'нет'}"
            detail = f"{providers} · {detail}" if detail else providers
        self._voice_detail_var.set(detail)
        listening = voice.active
        # Listening needs a working recogniser, speaking needs a working voice:
        # one being available never enables the other.
        self._voice_listen_button.configure(
            state="disabled" if listening or not voice.stt_provider else "normal"
        )
        self._voice_stop_button.configure(state="normal" if listening else "disabled")
        self._voice_speak_button.configure(state="normal" if voice.tts_provider else "disabled")

    def _start_voice(self) -> None:
        """Explicit action: listen once. Nothing records by itself."""
        if not self._bridge.start_voice():
            self._hint_var.set("голос недоступен или уже занят")

    def _cancel_voice(self) -> None:
        if self._bridge.cancel_voice():
            self._hint_var.set("запрошена отмена голосовой операции")
        else:
            self._hint_var.set("нечего отменять")

    def _speak_last_answer(self) -> None:
        last = ""
        for message in reversed(tuple(self._bridge.snapshot().messages)):
            if message.role == "assistant" and message.text.strip():
                last = message.text.strip()
                break
        if not last:
            self._hint_var.set("нечего озвучивать")
            return
        if self._bridge.speak(last):
            self._hint_var.set("ответ озвучен")
        else:
            self._hint_var.set("озвучивание недоступно")

    def _cancel_turn(self) -> None:
        if self._bridge.cancel_turn():
            self._hint_var.set(
                "Отмена запрошена: ход остановится на следующем безопасном шаге."
            )
        else:
            self._hint_var.set("Отменять нечего: ход не выполняется.")

    @staticmethod
    def _limits_text(view: StateView) -> str:
        limit = view.limits.get("max_tool_calls_per_message")
        text_channel = view.limits.get("allow_text_tool_calls")
        return (
            f"лимит вызовов: {limit}; текстовый канал: "
            f"{'включён' if text_channel else 'выключен'}"
        )

    def _render_transcript(self, view: StateView) -> None:
        if len(view.messages) == self._rendered:
            return
        self._rendered = len(view.messages)
        self._transcript.configure(state="normal")
        self._transcript.delete("1.0", "end")
        if not view.messages:
            self._transcript.insert("1.0", WELCOME, ("system",))
        for message in view.messages:
            if message.role == "user":
                tag, prefix = "user", "Вы: "
            elif message.role == "assistant":
                tag, prefix = "assistant", "Robanayt: "
            else:
                tag = "error" if message.meta == "ошибка" else "system"
                prefix = "Ошибка: " if tag == "error" else "— "
            self._transcript.insert("end", prefix, (tag,))
            self._transcript.insert("end", f"{message.text}\n\n", (tag,))
        self._transcript.configure(state="disabled")
        self._transcript.see("end")

    def _render_pending(self, pending: tuple[PendingView, ...]) -> None:
        ids = tuple(item.pending_id for item in pending)
        if ids == self._pending_ids:
            return
        self._pending_ids = ids
        for child in self._pending_frame.winfo_children():
            child.destroy()
        self._pending_empty.configure(text="нет" if not pending else "")
        for item in pending:
            self._build_approval_card(item)

    def _build_approval_card(self, pending: PendingView) -> None:
        card = ttk.Frame(self._pending_frame, style="Panel.TFrame", padding=(8, 6))
        card.pack(fill="x", pady=4)
        label(card, pending.tool, style_name="Section.TLabel").pack(anchor="w")
        label(
            card,
            f"v{pending.version} · {pending.reason or 'причина не указана'}",
            style_name="Panel.Dim.TLabel",
            wraplength=self._theme.metrics.inspector - 60,
        ).pack(anchor="w")
        actions = ttk.Frame(card, style="Panel.TFrame")
        actions.pack(fill="x", pady=(6, 0))
        button(actions, "Подтвердить", lambda item=pending: self._approve(item), accent=True).pack(
            side="left"
        )
        button(actions, "Отклонить", lambda item=pending: self._cancel(item), danger=True).pack(
            side="left", padx=(6, 0)
        )

    def _render_tools(self, view: StateView) -> None:
        current = tuple(self._tools_box.get(0, "end"))
        wanted = tuple(f"{name} — {permission}" for name, permission in view.tools)
        if current == wanted:
            return
        self._tools_box.delete(0, "end")
        for line in wanted:
            self._tools_box.insert("end", line)

    def _refresh_history(self, *, reset: bool = True) -> None:
        """Load one page of history. `reset=False` appends the next page."""
        if reset:
            self._history.delete(0, "end")
            self._history_ids = []
            self._history_offset = 0
        page = self._bridge.conversations(limit=self._page_size, offset=self._history_offset)
        for conversation_id, title in page:
            self._history_ids.append(conversation_id)
            self._history.insert("end", f"{title or 'без названия'} · {conversation_id[:8]}")
        self._history_offset += len(page)
        # "More" stays available only while the last page was full: no endless
        # scrolling through a history that has already ended.
        self._more_button.configure(
            state="normal" if len(page) >= self._page_size else "disabled"
        )

    # -- actions ------------------------------------------------------------

    def _send(self) -> None:
        text = self._input.get("1.0", "end-1c").strip()
        if not text:
            self._hint_var.set("Пустой запрос не отправляется.")
            return
        self._input.delete("1.0", "end")
        self._hint_var.set("Enter — отправить, Shift+Enter — новая строка")
        self._bridge.send(text)

    def _on_send(self, event: Any) -> str | None:
        if getattr(event, "state", 0) & 0x0001:  # Shift held: newline
            return None
        self._send()
        return "break"

    def _new_conversation(self) -> None:
        self._bridge.new_conversation()
        self._rendered = -1
        self._history.selection_clear(0, "end")

    def _on_select_conversation(self, _event: Any) -> None:
        selection = self._history.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index >= len(self._history_ids):
            return
        self._bridge.open_conversation(self._history_ids[index])
        self._rendered = -1

    def _approve(self, pending: PendingView) -> None:
        dialog = ApprovalDialog(self._root, pending, self._theme)
        if dialog.show():
            self._bridge.approve(pending.pending_id)

    def _cancel(self, pending: PendingView) -> None:
        self._bridge.cancel(pending.pending_id)

    def _open_settings(self) -> None:
        open_settings(self._root, self._bridge, self._theme, on_motion_change=self._set_motion)

    def _set_motion(self, reduced: bool) -> None:
        theme = REDUCED_MOTION_THEME if reduced else DEFAULT_THEME
        self._theme = theme
        apply_theme(self._root, theme)
        self._core.set_theme(theme)
        self._root.configure(background=theme.palette.bg)


def launch(bridge: DialogBridge, *, theme: Theme | None = None) -> None:
    """Build the window and run the main loop. Blocks until the window closes."""
    HolographicApp(bridge, theme=theme).run()
