"""Design tokens for the desktop UI.

One place for colours, typography, spacing and motion, so every panel looks
like a part of the same application. Dark graphite surfaces, a single cyan
accent, restrained glow - decoration never competes with the content.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from jarvis.surfaces.protocol import (
    STATE_ANSWERING,
    STATE_CANCELLED,
    STATE_CANCELLING,
    STATE_ERROR,
    STATE_IDLE,
    STATE_OFFLINE,
    STATE_THINKING,
    STATE_TOOL,
    STATE_WAITING,
)


@dataclass(frozen=True)
class Palette:
    """Colour tokens. Hex values only - tkinter understands them directly."""

    bg: str = "#05080f"
    surface: str = "#0b1220"
    surface_alt: str = "#111a2b"
    border: str = "#1d2b40"
    accent: str = "#38bdf8"
    accent_soft: str = "#0ea5e9"
    accent_dim: str = "#1d4f6b"
    text: str = "#e6f0fb"
    text_dim: str = "#8ba3bd"
    ok: str = "#34d399"
    warn: str = "#fbbf24"
    error: str = "#f87171"
    glow: str = "#22d3ee"


@dataclass(frozen=True)
class Typography:
    family: str = "Segoe UI"
    mono: str = "Consolas"
    small: int = 9
    body: int = 10
    title: int = 12
    heading: int = 16


@dataclass(frozen=True)
class Metrics:
    pad_x: int = 12
    pad_y: int = 8
    gap: int = 8
    sidebar: int = 250
    inspector: int = 300
    core: int = 190


@dataclass(frozen=True)
class Motion:
    """Durations in milliseconds. `core_tick_ms` caps the animation cost."""

    fast_ms: int = 120
    normal_ms: int = 220
    core_tick_ms: int = 70
    poll_ms: int = 150
    reduced: bool = False


@dataclass(frozen=True)
class Theme:
    name: str = "holographic-dark"
    palette: Palette = Palette()
    type: Typography = Typography()  # noqa: A003 - `type` reads well in tokens
    metrics: Metrics = Metrics()
    motion: Motion = Motion()


DEFAULT_THEME = Theme()

# Same colours, slower animation: for low-end machines and for users who ask
# for fewer effects. Colours and layout do not change.
REDUCED_MOTION_THEME = replace(
    DEFAULT_THEME,
    name="holographic-dark-reduced",
    motion=Motion(core_tick_ms=220, poll_ms=250, reduced=True),
)


def palette_for_state(state: str, theme: Theme = DEFAULT_THEME) -> str:
    """Accent colour for one application state. No state means no show."""
    mapping = {
        STATE_IDLE: theme.palette.accent_dim,
        STATE_THINKING: theme.palette.accent,
        STATE_ANSWERING: theme.palette.accent_soft,
        STATE_TOOL: theme.palette.glow,
        STATE_WAITING: theme.palette.warn,
        STATE_ERROR: theme.palette.error,
        STATE_OFFLINE: theme.palette.text_dim,
        STATE_CANCELLING: theme.palette.warn,
        STATE_CANCELLED: theme.palette.warn,
    }
    return mapping.get(state, theme.palette.accent_dim)
