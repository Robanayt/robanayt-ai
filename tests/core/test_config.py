"""Configuration: safe defaults, TOML loading, env overrides, loud failures."""

from __future__ import annotations

import pytest

from robanayt_ai.core.config import ConfigError, default_config, load_config


def test_defaults_are_safe_and_offline() -> None:
    config = default_config()

    assert config.source == "defaults"
    assert config.runtime.run_mode == "normal"
    assert config.runtime.single_instance is True
    assert config.logging.level == "INFO"
    assert config.bus.record_failures is True


def test_load_config_reads_toml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        '[logging]\nlevel = "debug"\n\n[runtime]\nrun_mode = "safe"\n',
        encoding="utf-8",
    )

    config = load_config()

    assert config.logging.level == "DEBUG"
    assert config.runtime.run_mode == "safe"
    assert config.source.endswith("config.toml")


def test_defaults_when_no_config_file_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.source == "defaults"


def test_explicit_missing_file_fails(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")


def test_unknown_key_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[runtime]\nbogus = 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_unknown_section_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[nope]\nmodel = \"x\"\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown configuration section"):
        load_config(path)


def test_wrong_value_type_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[runtime]\nsingle_instance = \"yes-please\"\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be true/false"):
        load_config(path)


def test_invalid_run_mode_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[runtime]\nrun_mode = "turbo"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown run mode"):
        load_config(path)


def test_invalid_log_level_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[logging]\nlevel = "LOUD"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid logging level"):
        load_config(path)


def test_environment_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_RUN_MODE", "diagnostic")
    monkeypatch.setenv("JARVIS_SINGLE_INSTANCE", "0")

    config = load_config()

    assert config.runtime.run_mode == "diagnostic"
    assert config.runtime.single_instance is False


def test_environment_bool_must_be_parseable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_LOG_TO_FILE", "maybe")

    with pytest.raises(ConfigError, match="JARVIS_LOG_TO_FILE"):
        load_config()


def test_describe_contains_no_secrets() -> None:
    summary = default_config().describe()

    assert set(summary) == {
        "source",
        "run_mode",
        "log_level",
        "single_instance",
        "log_file",
        "ai_enabled",
        "ai_provider",
        "ai_model",
        "tools_enabled",
        "orchestration_enabled",
        "voice_enabled",
    }
    assert summary["run_mode"] == "normal"
    assert "api_key" not in summary
