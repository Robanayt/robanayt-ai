"""The central visual core: geometry, glow and one honest state colour.

Animation cost is capped: a few arcs redrawn on a timer, and nothing moves
when the state is `idle` and motion is reduced.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any

from robanayt_ai.surfaces.protocol import (
    STATE_IDLE,
    STATE_LABELS,
    STATE_THINKING,
    STATE_WAITING,
)
from robanayt_ai.surfaces.theme import Theme, palette_for_state

_SPIN = {
    "idle": 0.35,
    "thinking": 2.6,
    "answering": 2.0,
    "tool": 3.2,
    "waiting": 1.2,
    "error": 0.0,
    "offline": 0.0,
    "cancelling": 1.2,
    "cancelled": 0.0,
}


class CoreWidget(tk.Canvas):
    """Concentric rings plus a centre dot; colour and speed follow the state."""

    def __init__(self, master: tk.Misc, theme: Theme, *, size: int | None = None) -> None:
        self._theme = theme
        self._size = size or theme.metrics.core
        super().__init__(
            master,
            width=self._size,
            height=self._size,
            background=theme.palette.bg,
            highlightthickness=0,
            borderwidth=0,
        )
        self._state = STATE_IDLE
        self._phase = 0.0
        self._alive = True
        self._job: str | None = None
        self.bind("<Destroy>", self._on_destroy)

    # -- public API ---------------------------------------------------------

    def set_state(self, state: str) -> None:
        self._state = state

    def set_theme(self, theme: Theme) -> None:
        """Switch tokens without rebuilding the widget."""
        self._theme = theme
        try:
            self.configure(
                background=theme.palette.bg,
                width=theme.metrics.core,
                height=theme.metrics.core,
            )
        except tk.TclError:  # pragma: no cover - widget already gone
            pass

    def start(self) -> None:
        self._alive = True
        self._tick()

    def stop(self) -> None:
        self._alive = False
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:  # pragma: no cover - widget already gone
                pass
            self._job = None

    # -- internals ----------------------------------------------------------

    def _on_destroy(self, _event: Any) -> None:
        self.stop()

    def _tick(self) -> None:
        if not self._alive:
            return
        self._phase = (self._phase + self._speed()) % 360.0
        self._draw()
        self._job = self.after(self._theme.motion.core_tick_ms, self._tick)

    def _speed(self) -> float:
        if self._theme.motion.reduced:
            return 0.0
        return _SPIN.get(self._state, 0.35)

    def _draw(self) -> None:
        self.delete("all")
        palette = self._theme.palette
        accent = palette_for_state(self._state, self._theme)
        centre = self._size / 2

        # Static guide ring: the geometry the moving arcs belong to.
        self.create_oval(
            centre - self._size * 0.46,
            centre - self._size * 0.46,
            centre + self._size * 0.46,
            centre + self._size * 0.46,
            outline=palette.border,
            width=1,
        )

        rings = (
            (0.40, 300.0, 2, palette.accent_dim),
            (0.31, 190.0, 2, palette.accent_soft),
            (0.22, 120.0, 2, accent),
        )
        for index, (ratio, extent, width, colour) in enumerate(rings):
            radius = self._size * ratio
            start = self._phase * (1 if index % 2 == 0 else -1) + 30 * index
            self.create_arc(
                centre - radius,
                centre - radius,
                centre + radius,
                centre + radius,
                start=start,
                extent=extent,
                style="arc",
                outline=colour,
                width=width,
            )

        # Centre dot, breathing only while the process is actually busy.
        busy = self._state in {STATE_THINKING, STATE_WAITING, "tool", "answering"}
        pulse = 0.0
        if busy and not self._theme.motion.reduced:
            pulse = math.sin(math.radians(self._phase * 3)) * 2.2
        core_radius = 6.0 + max(pulse, 0.0)
        self.create_oval(
            centre - core_radius,
            centre - core_radius,
            centre + core_radius,
            centre + core_radius,
            fill=accent,
            outline="",
        )
        if busy:
            halo = core_radius + 7.0
            self.create_oval(
                centre - halo,
                centre - halo,
                centre + halo,
                centre + halo,
                outline=palette.glow,
                width=1,
            )

        self.create_text(
            centre,
            self._size - 14,
            text=STATE_LABELS.get(self._state, self._state),
            fill=palette.text_dim,
            font=(self._theme.type.family, self._theme.type.small),
        )
