"""Tkinter implementation of the desktop UI.

Imported on demand: `import robanayt_ai` must never require a GUI toolkit.
"""

from __future__ import annotations

from robanayt_ai.surfaces.ui.window import HolographicApp, launch

__all__ = ["HolographicApp", "launch"]
