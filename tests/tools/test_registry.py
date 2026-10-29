"""Registry rules: uniqueness, lookup, explicit replacement and removal."""

from __future__ import annotations

import pytest

from tests.tools.helpers import (
    PermissionLevel,
    counting_handler,
    denied_spec,
    safe_spec,
)
from jarvis.tools.errors import DuplicateToolError, ToolNotFoundError, ToolSpecError
from jarvis.tools.models import ArgumentSpec, ToolContext
from jarvis.tools.registry import ToolRegistry


def noop(arguments: object, context: ToolContext) -> str:
    return "ok"


def test_register_and_get() -> None:
    registry = ToolRegistry()
    calls: list[dict] = []
    registry.register(safe_spec("ping"), counting_handler(calls))

    tool = registry.get("ping")
    assert tool is not None and tool.spec.name == "ping"
    assert registry.require("ping").spec.version == "1.0.0"


def test_registration_never_executes_the_handler() -> None:
    registry = ToolRegistry()
    calls: list[dict] = []
    registry.register(safe_spec("ping"), counting_handler(calls))
    assert calls == []


def test_lookup_returns_none_for_unknown_tool() -> None:
    registry = ToolRegistry()
    assert registry.get("missing") is None
    with pytest.raises(ToolNotFoundError):
        registry.require("missing")


def test_duplicate_names_are_rejected() -> None:
    registry = ToolRegistry()
    registry.register(safe_spec("ping"), noop)
    with pytest.raises(DuplicateToolError, match="already registered"):
        registry.register(safe_spec("ping"), noop)


def test_duplicate_can_be_replaced_explicitly() -> None:
    registry = ToolRegistry()
    registry.register(safe_spec("ping"), noop)
    registry.register(safe_spec("ping", version="2.0.0"), noop, replace=True)

    assert registry.require("ping").spec.version == "2.0.0"
    assert len(registry) == 1


def test_invalid_spec_is_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolSpecError, match="invalid tool spec"):
        registry.register("not-a-spec", noop)  # type: ignore[arg-type]


def test_non_callable_handler_is_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolSpecError, match="not callable"):
        registry.register(safe_spec("ping"), "not callable")  # type: ignore[arg-type]


def test_listing_is_sorted_by_name() -> None:
    registry = ToolRegistry()
    for name in ("zeta", "alpha", "mid"):
        registry.register(safe_spec(name), noop)

    assert registry.names() == ("alpha", "mid", "zeta")
    assert tuple(spec.name for spec in registry.specs()) == ("alpha", "mid", "zeta")
    assert "alpha" in registry


def test_describe_includes_arguments_and_permission() -> None:
    registry = ToolRegistry()
    registry.register(
        safe_spec("ping", arguments=(ArgumentSpec(name="a", kind="integer"),)),
        noop,
    )
    description = registry.describe()[0]
    assert description["name"] == "ping"
    assert description["permission"] == "safe"
    assert description["arguments"][0]["name"] == "a"


def test_unregister_is_explicit() -> None:
    registry = ToolRegistry()
    registry.register(safe_spec("ping"), noop)
    registry.unregister("ping")

    assert registry.get("ping") is None
    with pytest.raises(ToolNotFoundError):
        registry.unregister("ping")


def test_unregister_refuses_a_tool_in_use() -> None:
    registry = ToolRegistry()
    registry.register(safe_spec("ping"), noop)
    registry.mark_in_use("ping")

    with pytest.raises(ToolSpecError, match="in use"):
        registry.unregister("ping")
    assert registry.unregister("ping", force=True).spec.name == "ping"


def test_clear_requires_force() -> None:
    registry = ToolRegistry()
    registry.register(safe_spec("ping"), noop)
    registry.register(denied_spec(), noop)

    with pytest.raises(ToolSpecError, match="force"):
        registry.clear()
    assert registry.clear(force=True) == 2
    assert len(registry) == 0


def test_denied_tools_can_still_be_registered() -> None:
    registry = ToolRegistry()
    registry.register(denied_spec(), noop)
    assert registry.require("forbidden").spec.permission is PermissionLevel.DENIED
