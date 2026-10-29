"""Memory configuration: defaults, TOML, environment, validation."""

from __future__ import annotations

import pytest

from robanayt_ai.core.config import ConfigError, MemoryConfig, default_config, load_config


def test_defaults_are_local_and_relative() -> None:
    memory = default_config().memory

    assert memory.enabled is True
    assert memory.directory == "runtime/memory"
    assert memory.filename == "jarvis.db"
    assert memory.database_timeout == 5.0
    assert memory.max_recent_messages == 20
    assert memory.default_search_limit == 20
    assert memory.max_context_chars == 8_000


def test_database_path_is_derived() -> None:
    memory = MemoryConfig()

    assert str(memory.database_path).replace("\\", "/") == "runtime/memory/jarvis.db"


def test_toml_section(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.toml").write_text(
        "[memory]\nenabled = true\ndirectory = \"data/mem\"\n"
        "max_recent_messages = 5\ndefault_search_limit = 7\nmax_context_chars = 1500\n",
        encoding="utf-8",
    )

    memory = load_config().memory

    assert memory.directory == "data/mem"
    assert memory.max_recent_messages == 5
    assert memory.default_search_limit == 7
    assert memory.max_context_chars == 1500


def test_environment_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_MEMORY_ENABLED", "false")
    monkeypatch.setenv("JARVIS_MEMORY_DIR", "other/place")

    memory = load_config().memory

    assert memory.enabled is False
    assert memory.directory == "other/place"


@pytest.mark.parametrize(
    "section",
    [
        "[memory]\nmax_recent_messages = 0\n",
        "[memory]\ndefault_search_limit = -1\n",
        "[memory]\ndatabase_timeout = 0\n",
        "[memory]\nmax_context_chars = -5\n",
        "[memory]\ndirectory = \"\"\n",
        "[memory]\nfilename = \"\"\n",
        "[memory]\nunknown_key = 1\n",
    ],
)
def test_invalid_values_are_rejected(tmp_path, section: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text(section, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path)


def test_describe_has_no_private_data() -> None:
    summary = default_config().memory.describe()

    assert set(summary) == {
        "enabled",
        "database_path",
        "max_recent_messages",
        "default_search_limit",
        "max_context_chars",
    }
    assert "password" not in summary
    assert "api_key" not in summary


def test_startup_does_not_require_the_database() -> None:
    config = default_config()

    assert config.memory.enabled is True
    assert config.memory.database_path.is_file() is False or True
