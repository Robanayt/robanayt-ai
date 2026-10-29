"""Tkinter implementation of the desktop UI.

Imported on demand: `import jarvis` must never require a GUI toolkit.
"""

from __future__ import annotations

from jarvis.surfaces.ui.window import HolographicApp, launch

__all__ = ["HolographicApp", "launch"]
