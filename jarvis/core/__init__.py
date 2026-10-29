"""JARVIS_AI core layer.

Small, dependency-free foundation: configuration, in-process event bus,
explicit lifecycle, stdlib logging, run modes and a single-instance guard.

Nothing here starts AI, audio, GUI or automation - those arrive in later
phases and plug into this core.
"""

from jarvis.core.bus import APP_RUNNING, APP_STARTING, APP_STOPPED, APP_STOPPING, Event, EventBus
from jarvis.core.config import (
    BusConfig,
    Config,
    ConfigError,
    LoggingConfig,
    RuntimeConfig,
    load_config,
)
from jarvis.core.lifecycle import InvalidTransitionError, Lifecycle, State
from jarvis.core.log_config import setup_logging, shutdown_logging
from jarvis.core.run_mode import RunMode, RunModePolicy
from jarvis.core.single_instance import (
    InstanceAlreadyRunningError,
    LockOwner,
    SingleInstanceLock,
)

__all__ = [
    "APP_RUNNING",
    "APP_STARTING",
    "APP_STOPPED",
    "APP_STOPPING",
    "BusConfig",
    "Config",
    "ConfigError",
    "Event",
    "EventBus",
    "InstanceAlreadyRunningError",
    "InvalidTransitionError",
    "Lifecycle",
    "LockOwner",
    "LoggingConfig",
    "RunMode",
    "RunModePolicy",
    "RuntimeConfig",
    "SingleInstanceLock",
    "State",
    "load_config",
    "setup_logging",
    "shutdown_logging",
]
