"""AI bridge: descriptions out, untrusted calls in, never a permission."""

from __future__ import annotations

import pytest

from tests.tools.helpers import FakeClock
from robanayt_ai.core.config import ToolsConfig
from robanayt_ai.tools import build_tool_system
from robanayt_ai.tools.bridge import tool_call_from_ai, tool_descriptions
from robanayt_ai.tools.errors import ToolValidationError
from robanayt_ai.tools.models import ToolRequest, ToolStatus


def test_descriptions_are_provider_neutral() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())
    descriptions = tool_descriptions(system.registry)

    assert {item["name"] for item in descriptions} >= {"current_time", "add_numbers"}
    first = next(item for item in descriptions if item["name"] == "add_numbers")
    assert first["arguments"][0]["name"] == "a"
    assert "openai" not in str(descriptions).lower()


def test_valid_model_call_is_accepted() -> None:
    request = tool_call_from_ai({"name": "add_numbers", "arguments": {"a": 1, "b": 2}})

    assert isinstance(request, ToolRequest)
    assert request.tool == "add_numbers"
    assert request.arguments == {"a": 1, "b": 2}
    assert request.caller == "ai"


def test_call_without_arguments_is_accepted() -> None:
    request = tool_call_from_ai({"name": "current_time"})
    assert request.arguments == {}


@pytest.mark.parametrize("payload", ["string", 42, ["list"]])
def test_non_mapping_payload_is_rejected(payload: object) -> None:
    with pytest.raises(ToolValidationError, match="mapping"):
        tool_call_from_ai(payload)  # type: ignore[arg-type]


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ToolValidationError, match="unsupported tool call field"):
        tool_call_from_ai({"name": "ping", "extra": 1})


@pytest.mark.parametrize(
    "field", ["approved", "approval", "confirmed", "permission", "force", "bypass", "sudo"]
)
def test_model_cannot_grant_permission(field: str) -> None:
    with pytest.raises(ToolValidationError, match="cannot grant permission"):
        tool_call_from_ai({"name": "echo_text", field: True})


def test_missing_name_is_rejected() -> None:
    with pytest.raises(ToolValidationError, match="requires a non-empty 'name'"):
        tool_call_from_ai({"arguments": {}})


def test_arguments_must_be_a_mapping() -> None:
    with pytest.raises(ToolValidationError, match="must be a mapping"):
        tool_call_from_ai({"name": "ping", "arguments": [1, 2]})


def test_proposed_call_still_goes_through_the_policy() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())
    proposed = tool_call_from_ai({"name": "echo_text", "arguments": {"text": "hi"}})

    result = system.executor.execute(proposed)

    assert result.status is ToolStatus.APPROVAL_REQUIRED


def test_proposed_call_for_unknown_tool_is_not_executed() -> None:
    system = build_tool_system(ToolsConfig(), clock=FakeClock())
    proposed = tool_call_from_ai({"name": "not_a_tool", "arguments": {}})

    result = system.executor.execute(proposed)

    assert result.status is ToolStatus.NOT_FOUND
