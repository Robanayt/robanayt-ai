"""Explicit approval dialog: one call, its arguments, one decision."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from jarvis.surfaces.protocol import PendingView
from jarvis.surfaces.theme import Theme
from jarvis.surfaces.ui.widgets import (
    button,
    label,
    panel,
    read_only_text,
    scrolled_text,
)


class ApprovalDialog:
    """Modal card for one pending tool call.

    Shows the tool, its version and the exact arguments: approving is a
    conscious act. The approval itself is granted by the core
    (`DialogService.approve_pending`), never by this dialog.
    """

    def __init__(self, parent: tk.Misc, pending: PendingView, theme: Theme) -> None:
        self._pending = pending
        self._approved = False

        top = tk.Toplevel(parent)
        self._top = top
        top.title("Подтверждение действия")
        top.configure(background=theme.palette.bg)
        top.geometry("560x460")
        top.resizable(False, False)
        top.transient(parent)

        frame = panel(top, theme)
        frame.pack(fill="both", expand=True, padx=14, pady=14)

        label(frame, "Инструмент запрашивает разрешение", style_name="Title.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        for name, value in (
            ("Инструмент", pending.tool),
            ("Версия", pending.version),
            ("Идентификатор", pending.pending_id),
            ("Причина", pending.reason or "—"),
        ):
            row = ttk.Frame(frame, style="Panel.TFrame")
            row.pack(fill="x", pady=2)
            label(row, f"{name}:", style_name="Panel.Dim.TLabel", wraplength=None).pack(
                side="left", padx=(0, 6)
            )
            label(row, str(value), wraplength=420).pack(side="left")

        label(frame, "Аргументы вызова:", style_name="Section.TLabel").pack(
            anchor="w", pady=(10, 4)
        )
        _frame, text = scrolled_text(frame, theme, height=10, mono=True)
        text.insert("1.0", pending.arguments_text())
        read_only_text(text)
        _frame.pack(fill="both", expand=True)

        label(
            frame,
            "Разрешение действует один раз и привязано именно к этим аргументам.",
            style_name="Panel.Dim.TLabel",
            wraplength=480,
        ).pack(anchor="w", pady=(8, 4))

        actions = ttk.Frame(frame, style="Panel.TFrame")
        actions.pack(fill="x", pady=(6, 0))
        button(actions, "Отклонить", self._reject).pack(side="right", padx=(8, 0))
        button(actions, "Подтвердить", self._accept, accent=True).pack(side="right")

        top.grab_set()

    def _accept(self) -> None:
        self._approved = True
        self._top.destroy()

    def _reject(self) -> None:
        self._approved = False
        self._top.destroy()

    def show(self) -> bool:
        """Show the dialog and return the user's decision."""
        self._top.wait_window()
        return self._approved
