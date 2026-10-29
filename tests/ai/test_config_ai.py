"""AI configuration: safe defaults, TOML, env, validation, no secrets."""

from __future__ import annotations

import pytest

from robanayt_ai.core.config import AIConfig, ConfigError, default_config, load_config


def test_ai_defaults_require_nothing() -> None:
    ai = default_config().ai

    assert ai.enabled is True
    assert ai.default_provider == "ollama"
    assert ai.default_model == ""  # no model is assumed or downloaded
    assert ai.routing_policy == "local_first"
    assert ai.request_timeout == 30.0
    assert ai.max_retries == 0
    assert ai.ollama_base_url == "http://127.0.0.1:11434"
    assert ai.openai_base_url == ""
    assert ai.openai_api_key_env == "JARVIS_OPENAI_API_KEY"
    assert ai.allow_remote is False


def test_model_resolution_prefers_provider_specific_value() -> None:
    ai = AIConfig(default_model="shared", ollama_model="local-only")

    assert ai.resolve_ollama_model() == "local-only"
    assert ai.resolve_openai_model() == "shared"


def test_toml_ai_section(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        '[ai]\ndefault_provider = "openai_compatible"\ndefault_model = "gpt-test"\n'
        'request_timeout = 12.5\nmax_retries = 2\nallow_remote = true\n',
        encoding="utf-8",
    )

    ai = load_config().ai

    assert ai.default_provider == "openai_compatible"
    assert ai.default_model == "gpt-test"
    assert ai.request_timeout == 12.5
    assert ai.max_retries == 2
    assert ai.allow_remote is True


def test_env_overrides_ai(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_AI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("JARVIS_AI_MODEL", "env-model")
    monkeypatch.setenv("JARVIS_AI_ENABLED", "false")
    monkeypatch.setenv("JARVIS_OLLAMA_BASE_URL", "http://127.0.0.1:21434")

    ai = load_config().ai

    assert ai.default_provider == "openai_compatible"
    assert ai.default_model == "env-model"
    assert ai.enabled is False
    assert ai.ollama_base_url == "http://127.0.0.1:21434"


def test_invalid_routing_policy_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[ai]\nrouting_policy = "random"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid ai.routing_policy"):
        load_config(path)


def test_non_positive_timeout_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[ai]\nrequest_timeout = 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="request_timeout"):
        load_config(path)


def test_negative_retries_fail(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[ai]\nmax_retries = -1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="max_retries"):
        load_config(path)


def test_wrong_type_in_ai_section_fails(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[ai]\nenabled = \"yes\"\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be true/false"):
        load_config(path)


def test_api_key_value_is_never_a_config_field() -> None:
    field_names = {f.name for f in AIConfig.__dataclass_fields__.values()}

    assert "openai_api_key" not in field_names
    assert "api_key" not in field_names
    assert "openai_api_key_env" in field_names


def test_describe_contains_no_secrets() -> None:
    summary = default_config().describe()
    ai_summary = default_config().ai.describe()

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
    assert "api_key" not in ai_summary
    assert "secret" not in str(ai_summary).lower()
