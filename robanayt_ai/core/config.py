"""Typed configuration with safe defaults.

Rules for this layer:

- Everything has a default; JARVIS_AI starts with no configuration file.
- No network access, no AI model, no secrets and no API keys are involved.
- Unknown or mistyped keys fail loudly (`ConfigError`) instead of being
  silently ignored.
- Optional TOML file (`[logging]`, `[runtime]`, `[bus]`) plus a handful of
  `JARVIS_*` environment overrides is the whole surface.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, get_type_hints

from robanayt_ai.core.run_mode import RunMode

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
    tomllib = None  # type: ignore[assignment]

DEFAULT_CONFIG_PATH = Path("config") / "config.toml"

_VALID_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

# Upper bound for the bounded tool-call loop of one user message.
MAX_TOOL_CALLS_PER_MESSAGE = 10

_ENV_MAP: dict[str, tuple[str, str]] = {
    "JARVIS_RUN_MODE": ("runtime", "run_mode"),
    "JARVIS_SINGLE_INSTANCE": ("runtime", "single_instance"),
    "JARVIS_LOCK_PATH": ("runtime", "lock_path"),
    "JARVIS_LOG_LEVEL": ("logging", "level"),
    "JARVIS_LOG_TO_FILE": ("logging", "to_file"),
    # AI layer. The API key itself is never a configuration value - only the
    # name of the environment variable that holds it.
    "JARVIS_AI_ENABLED": ("ai", "enabled"),
    "JARVIS_AI_PROVIDER": ("ai", "default_provider"),
    "JARVIS_AI_MODEL": ("ai", "default_model"),
    "JARVIS_AI_ROUTING_POLICY": ("ai", "routing_policy"),
    "JARVIS_OLLAMA_BASE_URL": ("ai", "ollama_base_url"),
    "JARVIS_OPENAI_BASE_URL": ("ai", "openai_base_url"),
    "JARVIS_OPENAI_MODEL": ("ai", "openai_model"),
    # Memory layer. Only non-private settings are configurable; memory content
    # never appears in configuration.
    "JARVIS_MEMORY_ENABLED": ("memory", "enabled"),
    "JARVIS_MEMORY_DIR": ("memory", "directory"),
    # Tool layer. No tool is enabled by an environment variable: this only
    # controls the subsystem and its limits.
    "JARVIS_TOOLS_ENABLED": ("tools", "enabled"),
    "JARVIS_TOOLS_TIMEOUT": ("tools", "default_timeout_seconds"),
    "JARVIS_TOOLS_APPROVAL_TTL": ("tools", "approval_ttl_seconds"),
    "JARVIS_TOOLS_AUDIT": ("tools", "audit_enabled"),
    # Dialog orchestration. Controls the bounded turn cycle and its limits.
    "JARVIS_ORCHESTRATION_ENABLED": ("orchestration", "enabled"),
    "JARVIS_ORCHESTRATION_MAX_TOOL_CALLS": ("orchestration", "max_tool_calls_per_message"),
    "JARVIS_ORCHESTRATION_ALLOW_TEXT_CALLS": ("orchestration", "allow_text_tool_calls"),
    "JARVIS_ORCHESTRATION_PERSIST": ("orchestration", "persist_history"),
    # Voice layer (Phase 8). Nothing here enables automatic listening: every
    # capture or playback is an explicit user action.
    "JARVIS_VOICE_ENABLED": ("voice", "enabled"),
    "JARVIS_VOICE_STT": ("voice", "stt_provider"),
    "JARVIS_VOICE_TTS": ("voice", "tts_provider"),
    "JARVIS_VOICE_NAME": ("voice", "voice"),
    "JARVIS_VOICE_LANGUAGE": ("voice", "language"),
    "JARVIS_VOICE_PRIVACY": ("voice", "privacy"),
    "JARVIS_VOICE_TIMEOUT": ("voice", "listen_timeout_seconds"),
    "JARVIS_VOICE_SPEAK_RESPONSES": ("voice", "speak_responses"),
    "JARVIS_VOICE_MAX_CHARS": ("voice", "max_speech_chars"),
    "JARVIS_VOICE_ALLOW_REMOTE": ("voice", "allow_remote"),
}

_BOOL_ENV_KEYS = frozenset(
    {
        "single_instance",
        "to_file",
        "enabled",
        "allow_remote",
        "audit_enabled",
        "audit_to_file",
        "allow_text_tool_calls",
        "persist_history",
        "speak_responses",
    }
)
_FLOAT_ENV_KEYS = frozenset({"default_timeout_seconds", "approval_ttl_seconds", "listen_timeout_seconds"})
_INT_ENV_KEYS = frozenset({"max_tool_calls_per_message", "max_recent_messages", "max_speech_chars"})

# Kept in sync with robanayt_ai.ai.router.RoutingPolicy (core must not import ai).
# Kept in sync with robanayt_ai.voice.router privacy modes (core must not import voice).
_VALID_VOICE_PRIVACY = ("local_only", "any")

_VALID_ROUTING_POLICIES = (
    "local_first",
    "cloud_first",
    "fallback",
    "fastest",
    "cheapest",
    "strongest",
)


class ConfigError(RuntimeError):
    """Raised when configuration cannot be read or is invalid."""


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    console: bool = True
    to_file: bool = True
    directory: str = "runtime/logs"
    filename: str = "jarvis.log"

    @property
    def log_path(self) -> Path:
        return Path(self.directory) / self.filename


@dataclass(frozen=True)
class RuntimeConfig:
    run_mode: str = RunMode.NORMAL.value
    single_instance: bool = True
    lock_path: str = "runtime/jarvis.lock"
    data_dir: str = "runtime"


@dataclass(frozen=True)
class BusConfig:
    record_failures: bool = True
    max_recorded_failures: int = 50


@dataclass(frozen=True)
class AIConfig:
    """Intelligence layer settings.

    Defaults must keep JARVIS fully startable with no network, no Ollama,
    no API key and no downloaded model: AI is enabled, but no model is set,
    so providers report "not configured" instead of failing.
    """

    enabled: bool = True
    default_provider: str = "ollama"
    default_model: str = ""
    routing_policy: str = "local_first"
    request_timeout: float = 30.0
    max_retries: int = 0
    retry_delay_seconds: float = 0.0
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    openai_base_url: str = ""
    openai_model: str = ""
    openai_api_key_env: str = "JARVIS_OPENAI_API_KEY"
    openai_organization: str = ""
    allow_remote: bool = False

    def resolve_ollama_model(self) -> str:
        return self.ollama_model or self.default_model

    def resolve_openai_model(self) -> str:
        return self.openai_model or self.default_model

    def describe(self) -> dict[str, Any]:
        """Secrets-free summary (never includes an API key)."""
        return {
            "enabled": self.enabled,
            "default_provider": self.default_provider,
            "default_model": self.default_model or None,
            "routing_policy": self.routing_policy,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "ollama_base_url": self.ollama_base_url,
            "openai_base_url": self.openai_base_url or None,
            "allow_remote": self.allow_remote,
        }


@dataclass(frozen=True)
class MemoryConfig:
    """Persistent memory settings.

    Defaults stay local and relative: no absolute user-specific path, no
    requirement for the database to exist at startup (it is created lazily
    on first use).
    """

    enabled: bool = True
    directory: str = "runtime/memory"
    filename: str = "jarvis.db"
    database_timeout: float = 5.0
    max_recent_messages: int = 20
    default_search_limit: int = 20
    max_context_chars: int = 8_000
    export_directory: str = "runtime/exports"

    @property
    def database_path(self) -> Path:
        return Path(self.directory) / self.filename

    def describe(self) -> dict[str, Any]:
        """Diagnostics without private data."""
        return {
            "enabled": self.enabled,
            "database_path": str(self.database_path),
            "max_recent_messages": self.max_recent_messages,
            "default_search_limit": self.default_search_limit,
            "max_context_chars": self.max_context_chars,
        }


@dataclass(frozen=True)
class ToolsConfig:
    """Tool subsystem settings.

    Defaults are deliberately conservative: the subsystem is enabled (so the
    safe demo tools are available), unknown tools are denied by the policy,
    executions have a short timeout and the audit log stays in memory (no
    file is created at startup).
    """

    enabled: bool = True
    default_timeout_seconds: float = 5.0
    approval_ttl_seconds: float = 300.0
    audit_enabled: bool = True
    audit_to_file: bool = False
    audit_directory: str = "runtime/logs"
    audit_filename: str = "tools.jsonl"

    def describe(self) -> dict[str, Any]:
        """Diagnostics without private data."""
        return {
            "enabled": self.enabled,
            "default_timeout_seconds": self.default_timeout_seconds,
            "approval_ttl_seconds": self.approval_ttl_seconds,
            "audit_enabled": self.audit_enabled,
            "audit_to_file": self.audit_to_file,
        }


@dataclass(frozen=True)
class OrchestrationConfig:
    """Dialog orchestration settings.

    Defaults are deliberately small: at most three tool executions per user
    message, history persisted, and no tool calls guessed from free text.
    """

    enabled: bool = True
    max_tool_calls_per_message: int = 3
    allow_text_tool_calls: bool = False
    persist_history: bool = True
    pending_ttl_seconds: float = 300.0

    def describe(self) -> dict[str, Any]:
        """Diagnostics without private data."""
        return {
            "enabled": self.enabled,
            "max_tool_calls_per_message": self.max_tool_calls_per_message,
            "allow_text_tool_calls": self.allow_text_tool_calls,
            "persist_history": self.persist_history,
        }


@dataclass(frozen=True)
class VoiceConfig:
    """Voice subsystem settings (Phase 8).

    Defaults are deliberately conservative:

    - nothing speaks automatically (`speak_responses=false`);
    - nothing leaves the machine (`privacy=local_only`, `allow_remote=false`);
    - a provider may be named explicitly, or `auto` picks the first one that
      is really available, or `none` disables that side.

    A provider's availability is decided by probing the system, never by the
    presence of its class.
    """

    enabled: bool = True
    stt_provider: str = "auto"
    tts_provider: str = "auto"
    voice: str = ""
    language: str = ""
    privacy: str = "local_only"
    listen_timeout_seconds: float = 8.0
    speak_responses: bool = False
    max_speech_chars: int = 2000
    allow_remote: bool = False

    def describe(self) -> dict[str, Any]:
        """Diagnostics without audio data and without secrets."""
        return {
            "enabled": self.enabled,
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
            "voice": self.voice or None,
            "language": self.language or None,
            "privacy": self.privacy,
            "listen_timeout_seconds": self.listen_timeout_seconds,
            "speak_responses": self.speak_responses,
            "max_speech_chars": self.max_speech_chars,
            "allow_remote": self.allow_remote,
        }


@dataclass(frozen=True)
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    bus: BusConfig = field(default_factory=BusConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    source: str = "defaults"

    def describe(self) -> dict[str, Any]:
        """Compact summary for startup output and diagnostics."""
        return {
            "source": self.source,
            "run_mode": self.runtime.run_mode,
            "log_level": self.logging.level,
            "single_instance": self.runtime.single_instance,
            "log_file": str(self.logging.log_path) if self.logging.to_file else None,
            "ai_enabled": self.ai.enabled,
            "ai_provider": self.ai.default_provider,
            "ai_model": self.ai.default_model or None,
            "tools_enabled": self.tools.enabled,
            "orchestration_enabled": self.orchestration.enabled,
            "voice_enabled": self.voice.enabled,
        }


def default_config() -> Config:
    """Return the built-in defaults (no file, no environment)."""
    return Config()


def load_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Load configuration: defaults, then optional TOML file, then env."""
    env = os.environ if env is None else env
    data: dict[str, Any] = {}
    source = "defaults"

    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(f"configuration file not found: {config_path}")
    elif DEFAULT_CONFIG_PATH.is_file():
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = None

    if config_path is not None:
        data = _read_toml(config_path)
        source = str(config_path)

    config = Config(
        logging=_merge_section(LoggingConfig(), data.get("logging", {}), "logging"),
        runtime=_merge_section(RuntimeConfig(), data.get("runtime", {}), "runtime"),
        bus=_merge_section(BusConfig(), data.get("bus", {}), "bus"),
        ai=_merge_section(AIConfig(), data.get("ai", {}), "ai"),
        memory=_merge_section(MemoryConfig(), data.get("memory", {}), "memory"),
        tools=_merge_section(ToolsConfig(), data.get("tools", {}), "tools"),
        orchestration=_merge_section(
            OrchestrationConfig(), data.get("orchestration", {}), "orchestration"
        ),
        voice=_merge_section(VoiceConfig(), data.get("voice", {}), "voice"),
        source=source,
    )
    config = _apply_env(config, env)
    config = replace(
        config,
        logging=replace(config.logging, level=config.logging.level.strip().upper()),
    )
    _validate(config)
    return config


def _read_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:  # pragma: no cover - Python 3.10 only
        raise ConfigError(
            f"reading {path} requires Python 3.11+ (tomllib); "
            "run on a newer interpreter or use defaults"
        )
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:  # type: ignore[union-attr]
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"configuration root must be a table: {path}")

    unknown_sections = sorted(
        set(data)
        - {"logging", "runtime", "bus", "ai", "memory", "tools", "orchestration", "voice"}
    )
    if unknown_sections:
        raise ConfigError(
            f"unknown configuration section(s): {', '.join(unknown_sections)}; "
            "allowed: logging, runtime, bus, ai, memory, tools, orchestration, voice"
        )
    return data


def _merge_section(instance: Any, data: Any, section: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"[{section}] must be a table")

    allowed = {f.name for f in fields(type(instance))}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in [{section}]: {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )

    hints = get_type_hints(type(instance))
    updates: dict[str, Any] = {}
    for key, value in data.items():
        expected = hints.get(key)
        if expected is bool and not isinstance(value, bool):
            raise ConfigError(f"[{section}] {key} must be true/false, got {value!r}")
        if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
            raise ConfigError(f"[{section}] {key} must be an integer, got {value!r}")
        if expected is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ConfigError(f"[{section}] {key} must be a number, got {value!r}")
        if expected is str and not isinstance(value, str):
            raise ConfigError(f"[{section}] {key} must be a string, got {value!r}")
        updates[key] = float(value) if expected is float else value
    return replace(instance, **updates)


def _apply_env(config: Config, env: Mapping[str, str]) -> Config:
    overrides: dict[str, dict[str, Any]] = {}
    for name, (section, key) in _ENV_MAP.items():
        raw = env.get(name)
        if raw is None or raw == "":
            continue
        value: Any
        if key in _BOOL_ENV_KEYS:
            lowered = raw.strip().lower()
            if lowered in _TRUE_VALUES:
                value = True
            elif lowered in _FALSE_VALUES:
                value = False
            else:
                raise ConfigError(
                    f"{name} must be one of true/false/1/0/yes/no/on/off, got {raw!r}"
                )
        elif key in _FLOAT_ENV_KEYS:
            try:
                value = float(raw.strip())
            except ValueError as exc:
                raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
        elif key in _INT_ENV_KEYS:
            try:
                value = int(raw.strip())
            except ValueError as exc:
                raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
        else:
            value = raw
        overrides.setdefault(section, {})[key] = value

    sections = {
        "logging": config.logging,
        "runtime": config.runtime,
        "bus": config.bus,
        "ai": config.ai,
        "memory": config.memory,
        "tools": config.tools,
        "orchestration": config.orchestration,
        "voice": config.voice,
    }
    for section, values in overrides.items():
        sections[section] = _merge_section(sections[section], values, section)

    return replace(config, **sections)


def _validate(config: Config) -> None:
    if config.logging.level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"invalid logging level {config.logging.level!r}; "
            f"allowed: {', '.join(_VALID_LOG_LEVELS)}"
        )
    try:
        RunMode.from_value(config.runtime.run_mode)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    if config.bus.max_recorded_failures < 0:
        raise ConfigError("bus.max_recorded_failures must be >= 0")

    ai = config.ai
    if ai.routing_policy not in _VALID_ROUTING_POLICIES:
        raise ConfigError(
            f"invalid ai.routing_policy {ai.routing_policy!r}; "
            f"allowed: {', '.join(_VALID_ROUTING_POLICIES)}"
        )
    if ai.request_timeout <= 0:
        raise ConfigError("ai.request_timeout must be > 0")
    if ai.max_retries < 0:
        raise ConfigError("ai.max_retries must be >= 0")
    if ai.retry_delay_seconds < 0:
        raise ConfigError("ai.retry_delay_seconds must be >= 0")
    if not ai.default_provider.strip():
        raise ConfigError("ai.default_provider must not be empty")
    if not ai.openai_api_key_env.strip():
        raise ConfigError("ai.openai_api_key_env must not be empty")

    memory = config.memory
    if not memory.directory.strip():
        raise ConfigError("memory.directory must not be empty")
    if not memory.filename.strip():
        raise ConfigError("memory.filename must not be empty")
    if memory.database_timeout <= 0:
        raise ConfigError("memory.database_timeout must be > 0")
    if memory.max_recent_messages < 1:
        raise ConfigError("memory.max_recent_messages must be >= 1")
    if memory.default_search_limit < 1:
        raise ConfigError("memory.default_search_limit must be >= 1")
    if memory.max_context_chars < 0:
        raise ConfigError("memory.max_context_chars must be >= 0 (0 means unlimited)")
    if not memory.export_directory.strip():
        raise ConfigError("memory.export_directory must not be empty")

    tools = config.tools
    if tools.default_timeout_seconds <= 0:
        raise ConfigError("tools.default_timeout_seconds must be > 0")
    if tools.approval_ttl_seconds <= 0:
        raise ConfigError("tools.approval_ttl_seconds must be > 0")
    if tools.audit_to_file and not tools.audit_directory.strip():
        raise ConfigError("tools.audit_directory must not be empty when audit_to_file=true")
    if not tools.audit_filename.strip():
        raise ConfigError("tools.audit_filename must not be empty")

    orchestration = config.orchestration
    if orchestration.max_tool_calls_per_message < 0:
        raise ConfigError("orchestration.max_tool_calls_per_message must be >= 0")
    if orchestration.max_tool_calls_per_message > MAX_TOOL_CALLS_PER_MESSAGE:
        raise ConfigError(
            "orchestration.max_tool_calls_per_message must be <= "
            f"{MAX_TOOL_CALLS_PER_MESSAGE}"
        )
    if orchestration.pending_ttl_seconds <= 0:
        raise ConfigError("orchestration.pending_ttl_seconds must be > 0")

    voice = config.voice
    if voice.privacy not in _VALID_VOICE_PRIVACY:
        raise ConfigError(
            f"invalid voice.privacy {voice.privacy!r}; "
            f"allowed: {', '.join(_VALID_VOICE_PRIVACY)}"
        )
    if voice.listen_timeout_seconds <= 0:
        raise ConfigError("voice.listen_timeout_seconds must be > 0")
    if voice.max_speech_chars < 1:
        raise ConfigError("voice.max_speech_chars must be >= 1")
    if not voice.stt_provider.strip():
        raise ConfigError("voice.stt_provider must not be empty")
    if not voice.tts_provider.strip():
        raise ConfigError("voice.tts_provider must not be empty")

    logging.getLogger(__name__).debug("configuration validated: %s", config.source)
