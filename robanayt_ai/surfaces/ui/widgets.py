"""Themed tkinter helpers: one place that knows how a control looks."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from robanayt_ai.surfaces.theme import Theme


def apply_theme(root: tk.Misc, theme: Theme) -> ttk.Style:
    """Configure ttk so every widget follows the same tokens."""
    palette = theme.palette
    fonts = theme.type
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:  # pragma: no cover - depends on the Tk build
        pass

    root.configure(background=palette.bg)
    style.configure(
        ".",
        background=palette.surface,
        foreground=palette.text,
        fieldbackground=palette.surface_alt,
        bordercolor=palette.border,
        font=(fonts.family, fonts.body),
    )
    style.configure("TFrame", background=palette.surface)
    style.configure("Panel.TFrame", background=palette.surface_alt, relief="flat")
    style.configure("TLabel", background=palette.surface, foreground=palette.text)
    style.configure("Panel.TLabel", background=palette.surface_alt, foreground=palette.text)
    style.configure("Dim.TLabel", foreground=palette.text_dim)
    style.configure("Panel.Dim.TLabel", background=palette.surface_alt, foreground=palette.text_dim)
    style.configure("Title.TLabel", font=(fonts.family, fonts.heading, "bold"), foreground=palette.text)
    style.configure("Section.TLabel", font=(fonts.family, fonts.title, "bold"))
    style.configure(
        "TButton",
        background=palette.surface_alt,
        foreground=palette.text,
        bordercolor=palette.border,
        focuscolor=palette.accent_dim,
        padding=(10, 6),
    )
    style.map(
        "TButton",
        background=[("active", palette.accent_dim), ("disabled", palette.surface)],
        foreground=[("disabled", palette.text_dim)],
    )
    style.configure("Accent.TButton", background=palette.accent_dim, foreground=palette.text)
    style.map("Accent.TButton", background=[("active", palette.accent_soft)])
    style.configure("Danger.TButton", background=palette.surface_alt, foreground=palette.error)
    style.configure(
        "TCheckbutton",
        background=palette.surface,
        foreground=palette.text,
        focuscolor=palette.accent_dim,
    )
    style.configure("TNotebook", background=palette.bg, bordercolor=palette.border)
    style.configure(
        "TNotebook.Tab",
        background=palette.surface,
        foreground=palette.text_dim,
        padding=(12, 6),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", palette.surface_alt)],
        foreground=[("selected", palette.text)],
    )
    style.configure(
        "Vertical.TScrollbar",
        background=palette.surface_alt,
        troughcolor=palette.bg,
        bordercolor=palette.border,
        arrowcolor=palette.text_dim,
    )
    return style


def panel(parent: tk.Misc, theme: Theme, padding: tuple[int, int] = (12, 10)) -> ttk.Frame:
    return ttk.Frame(parent, style="Panel.TFrame", padding=padding)


def label(
    parent: tk.Misc,
    text: str = "",
    *,
    style_name: str = "Panel.TLabel",
    wraplength: int | None = None,
    textvariable: tk.Variable | None = None,
) -> ttk.Label:
    return ttk.Label(
        parent,
        text=text,
        textvariable=textvariable,
        style=style_name,
        wraplength=wraplength,
    )


def button(
    parent: tk.Misc,
    text: str,
    command: Any,
    *,
    accent: bool = False,
    danger: bool = False,
) -> ttk.Button:
    style_name = "Accent.TButton" if accent else "Danger.TButton" if danger else "TButton"
    return ttk.Button(parent, text=text, command=command, style=style_name)


def scrolled_text(
    parent: tk.Misc,
    theme: Theme,
    *,
    height: int = 12,
    mono: bool = False,
) -> tuple[ttk.Frame, tk.Text]:
    """A frame with a text widget and its scrollbar. Returns (frame, text)."""
    palette = theme.palette
    frame = ttk.Frame(parent, style="Panel.TFrame")
    family = theme.type.mono if mono else theme.type.family
    widget = tk.Text(
        frame,
        height=height,
        wrap="word",
        background=palette.surface,
        foreground=palette.text,
        insertbackground=palette.accent,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=palette.border,
        font=(family, theme.type.body),
        padx=10,
        pady=8,
    )
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
    widget.configure(yscrollcommand=scrollbar.set)
    widget.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return frame, widget


def read_only_text(text: tk.Text) -> None:
    text.configure(state="disabled")


def write_text(text: tk.Text, content: str) -> None:
    """Replace the whole content, keeping the widget read-only for the user."""
    text.configure(state="normal")
    text.delete("1.0", "end")
    text.insert("1.0", content)
    text.configure(state="disabled")
