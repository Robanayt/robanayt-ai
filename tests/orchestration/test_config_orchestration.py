"""Orchestration configuration: safe defaults, TOML, env and validation."""

from __future__ import annotations

import pytest

from robanayt_ai.core.config import ConfigError, OrchestrationConfig, default_config, load_config


def test_defaults_are_bounded() -> None:
    config = default_config().orchestration

    assert config.enabled is True
    assert config.max_tool_calls_per_message == 3
    assert config.allow_text_tool_calls is False  # no guessing from prose
    assert config.persist_history is True
    assert config.pending_ttl_seconds == 300.0


def test_describe_is_compact() -> None:
    assert OrchestrationConfig().describe() == {
        "enabled": True,
        "max_tool_calls_per_message": 3,
        "allow_text_tool_calls": False,
        "persist_history": True,
    }


def test_toml_section_is_loaded(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        "[orchestration]\nenabled = false\nmax_tool_calls_per_message = 5\n"
        "allow_text_tool_calls = true\npersist_history = false\n",
        encoding="utf-8",
    )

    config = load_config().orchestration

    assert config.enabled is False
    assert config.max_tool_calls_per_message == 5
    assert config.allow_text_tool_calls is True
    assert config.persist_history is False


def test_unknown_key_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[orchestration]\nbogus = 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_env_overrides() -> None:
    assert (
        load_config(env={"JARVIS_ORCHESTRATION_ENABLED": "false"}).orchestration.enabled
        is False
    )
    assert (
        load_config(env={"JARVIS_ORCHESTRATION_MAX_TOOL_CALLS": "1"})
        .orchestration.max_tool_calls_per_message
        == 1
    )
    assert (
        load_config(env={"JARVIS_ORCHESTRATION_ALLOW_TEXT_CALLS": "yes"})
        .orchestration.allow_text_tool_calls
        is True
    )
    assert (
        load_config(env={"JARVIS_ORCHESTRATION_PERSIST": "off"})
        .orchestration.persist_history
        is False
    )


def test_bad_integer_env_fails() -> None:
    with pytest.raises(ConfigError, match="must be an integer"):
        load_config(env={"JARVIS_ORCHESTRATION_MAX_TOOL_CALLS": "many"})


def test_negative_limit_is_rejected(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[orchestration]\nmax_tool_calls_per_message = -1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be >= 0"):
        load_config(path)


def test_excessive_limit_is_rejected(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[orchestration]\nmax_tool_calls_per_message = 500\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be <="):
        load_config(path)


def test_upper_bound_is_accepted(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[orchestration]\nmax_tool_calls_per_message = 10\n", encoding="utf-8")

    assert load_config(path).orchestration.max_tool_calls_per_message == 10


def test_non_positive_pending_ttl_is_rejected(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[orchestration]\npending_ttl_seconds = 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="pending_ttl_seconds must be > 0"):
        load_config(path)
