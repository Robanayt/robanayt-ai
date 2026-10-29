"""Tool configuration: safe defaults, TOML, env and validation."""

from __future__ import annotations

import pytest

from jarvis.core.config import ConfigError, ToolsConfig, default_config, load_config


def test_defaults_are_safe_and_local() -> None:
    config = default_config().tools

    assert config.enabled is True
    assert config.default_timeout_seconds == 5.0
    assert config.approval_ttl_seconds == 300.0
    assert config.audit_enabled is True
    assert config.audit_to_file is False  # no file is created at startup


def test_describe_has_no_private_data() -> None:
    summary = ToolsConfig().describe()

    assert summary == {
        "enabled": True,
        "default_timeout_seconds": 5.0,
        "approval_ttl_seconds": 300.0,
        "audit_enabled": True,
        "audit_to_file": False,
    }


def test_toml_section_is_loaded(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        '[tools]\nenabled = false\ndefault_timeout_seconds = 1.5\napproval_ttl_seconds = 30.0\n',
        encoding="utf-8",
    )

    config = load_config().tools

    assert config.enabled is False
    assert config.default_timeout_seconds == 1.5
    assert config.approval_ttl_seconds == 30.0


def test_unknown_tool_key_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[tools]\nbogus = 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_boolean_env_overrides() -> None:
    assert load_config(env={"JARVIS_TOOLS_ENABLED": "false"}).tools.enabled is False
    assert load_config(env={"JARVIS_TOOLS_AUDIT": "off"}).tools.audit_enabled is False
    assert load_config(env={"JARVIS_TOOLS_AUDIT": "no"}).tools.audit_enabled is False


def test_float_env_override_is_parsed(monkeypatch) -> None:
    config = load_config(env={"JARVIS_TOOLS_TIMEOUT": "2.5"}).tools
    assert config.default_timeout_seconds == 2.5


def test_bad_float_env_fails() -> None:
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(env={"JARVIS_TOOLS_TIMEOUT": "soon"})


def test_bad_boolean_env_fails() -> None:
    with pytest.raises(ConfigError, match="true/false"):
        load_config(env={"JARVIS_TOOLS_ENABLED": "maybe"})


def test_non_positive_timeout_is_rejected(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[tools]\ndefault_timeout_seconds = 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="default_timeout_seconds must be > 0"):
        load_config(path)


def test_non_positive_ttl_is_rejected(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[tools]\napproval_ttl_seconds = -1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="approval_ttl_seconds must be > 0"):
        load_config(path)


def test_file_audit_requires_a_directory(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[tools]\naudit_to_file = true\naudit_directory = ""\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="audit_directory"):
        load_config(path)


def test_empty_audit_filename_is_rejected(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[tools]\naudit_filename = ""\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="audit_filename"):
        load_config(path)


def test_disabled_tools_keep_startup_working() -> None:
    config = load_config(env={"JARVIS_TOOLS_ENABLED": "false"})

    assert config.tools.enabled is False
    assert config.runtime.run_mode == "normal"
