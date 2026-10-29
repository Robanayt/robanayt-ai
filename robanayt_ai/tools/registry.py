"""Tool registry (Phase 4).

Registration is data, not execution: adding a tool never calls its handler
and never touches anything outside this process.

Replacement and removal are deliberately explicit:

- `register()` rejects duplicates unless `replace=True`;
- `unregister()` refuses to remove a tool that a live executor has marked as
  in use, unless `force=True` - a running system must not silently lose a
  tool.
"""

from __future__ import annotations

import logging
from typing import Any

from robanayt_ai.tools.errors import DuplicateToolError, ToolNotFoundError, ToolSpecError
from robanayt_ai.tools.models import Tool, ToolHandler, ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Name -> tool mapping with strict, predictable rules."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._in_use: set[str] = set()

    # -- registration ----------------------------------------------------

    def register(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        replace: bool = False,
    ) -> Tool:
        """Register a tool. Never executes it."""
        if not isinstance(spec, ToolSpec):
            raise ToolSpecError(f"invalid tool spec: {type(spec).__name__}")
        if not callable(handler):
            raise ToolSpecError(f"handler for {spec.name!r} is not callable")
        if spec.name in self._tools and not replace:
            raise DuplicateToolError(
                f"tool {spec.name!r} is already registered; "
                "pass replace=True to override it explicitly"
            )
        tool = Tool(spec=spec, handler=handler)
        self._tools[spec.name] = tool
        logger.debug("tool registered: %s (version=%s)", spec.name, spec.version)
        return tool

    def unregister(self, name: str, *, force: bool = False) -> Tool:
        """Remove a tool explicitly. A missing tool raises."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"tool {name!r} is not registered")
        if name in self._in_use and not force:
            raise ToolSpecError(
                f"tool {name!r} is in use; pass force=True to remove it anyway"
            )
        del self._tools[name]
        self._in_use.discard(name)
        logger.debug("tool unregistered: %s", name)
        return tool

    def clear(self, *, force: bool = False) -> int:
        """Drop every tool. Returns the number removed."""
        if not force:
            raise ToolSpecError("clearing the registry requires force=True")
        removed = len(self._tools)
        self._tools.clear()
        self._in_use.clear()
        return removed

    # -- lookup ----------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the tool or None - lookup never raises."""
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self.get(name)
        if tool is None:
            raise ToolNotFoundError(f"tool {name!r} is not registered")
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools[name].spec for name in self.names())

    def list(self) -> tuple[Tool, ...]:
        return tuple(self._tools[name] for name in self.names())

    def describe(self) -> list[dict[str, Any]]:
        """Provider-neutral descriptions, sorted by name."""
        return [tool.spec.describe() for tool in self.list()]

    # -- usage bookkeeping ------------------------------------------------

    def mark_in_use(self, name: str) -> None:
        self._in_use.add(name)

    def release_in_use(self, name: str) -> None:
        self._in_use.discard(name)

    def in_use(self) -> tuple[str, ...]:
        return tuple(sorted(self._in_use))

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools
