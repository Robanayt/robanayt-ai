"""Run modes: parsing and capability policies."""

from __future__ import annotations

import pytest

from robanayt_ai.core.run_mode import RunMode


def test_from_value_accepts_known_modes() -> None:
    assert RunMode.from_value("normal") is RunMode.NORMAL
    assert RunMode.from_value("SAFE") is RunMode.SAFE
    assert RunMode.from_value(" diagnostic ") is RunMode.DIAGNOSTIC
    assert RunMode.from_value(RunMode.DEVELOPMENT) is RunMode.DEVELOPMENT


def test_none_defaults_to_normal() -> None:
    assert RunMode.from_value(None) is RunMode.NORMAL


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown run mode"):
        RunMode.from_value("turbo")


def test_no_mode_permits_network() -> None:
    assert all(mode.policy.allow_network is False for mode in RunMode)


def test_safe_mode_disables_tools() -> None:
    assert RunMode.SAFE.policy.allow_tools is False
    assert RunMode.NORMAL.policy.allow_tools is True


def test_verbose_modes_use_debug_logging() -> None:
    assert RunMode.DIAGNOSTIC.policy.log_level == "DEBUG"
    assert RunMode.DEVELOPMENT.policy.log_level == "DEBUG"
    assert RunMode.NORMAL.policy.log_level == "INFO"


def test_mock_backends_only_in_verbose_modes() -> None:
    assert RunMode.NORMAL.policy.mock_backends is False
    assert RunMode.SAFE.policy.mock_backends is False
    assert RunMode.DIAGNOSTIC.policy.mock_backends is True
    assert RunMode.DEVELOPMENT.policy.mock_backends is True


def test_describe_exposes_policy() -> None:
    summary = RunMode.SAFE.describe()

    assert summary["run_mode"] == "safe"
    assert summary["allow_tools"] is False
    assert summary["allow_network"] is False
