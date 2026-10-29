"""Provider-independent tool data models (Phase 4).

Everything here is plain data: no provider SDK, no network, no dynamic code
execution. Validation is explicit and strict:

- a tool name is a short lowercase identifier;
- every argument is described by an `ArgumentSpec` and validated before the
  handler is ever called;
- a request carries a stable digest of its arguments, so an approval can be
  bound to the exact action the user agreed to;
- a result is always a `ToolResult` - callers never see a raw handler value
  and never have to guess whether the action really happened.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 2_000
MAX_ARGUMENTS = 32
MAX_VERSION_CHARS = 32

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_VERSION_PATTERN = re.compile(r"^[0-9]+(\.[0-9]+){0,2}$")

_ARGUMENT_KINDS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
}


class ToolStatus(str, Enum):
    """Outcome of one execution attempt.

    Only `SUCCESS` means the action happened. `APPROVAL_REQUIRED`,
    `DENIED`, `INVALID`, `NOT_FOUND`, `TIMEOUT` and `FAILED` all mean the
    tool did not produce a trustworthy result.
    """

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"
    INVALID = "invalid"
    NOT_FOUND = "not_found"


class PermissionLevel(str, Enum):
    """How a tool may be executed.

    - `SAFE`: no external side effects, may run immediately.
    - `APPROVAL`: needs a valid approval bound to these exact arguments.
    - `DENIED`: never executed, whatever the caller (or a model) says.
    """

    SAFE = "safe"
    APPROVAL = "approval_required"
    DENIED = "denied"


def new_id(prefix: str) -> str:
    """Short unique identifier - readable in logs, not security-relevant."""
    return f"{prefix}_{hashlib.sha256(f'{prefix}{time.time_ns()}{id(object())}'.encode()).hexdigest()[:16]}"


def validate_name(value: str, what: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    name = value.strip()
    if len(name) > MAX_NAME_CHARS:
        raise ValueError(f"{what} must be at most {MAX_NAME_CHARS} characters")
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            f"{what} {name!r} is invalid: use lowercase letters, digits and _ . : -"
        )
    return name


def validate_description(value: str, what: str = "description") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    text = value.strip()
    if len(text) > MAX_DESCRIPTION_CHARS:
        raise ValueError(f"{what} must be at most {MAX_DESCRIPTION_CHARS} characters")
    return text


def validate_version(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("version must be a non-empty string")
    version = value.strip()
    if len(version) > MAX_VERSION_CHARS or not _VERSION_PATTERN.match(version):
        raise ValueError(f"version {version!r} must look like 1, 1.2 or 1.2.3")
    return version


def _canonical(value: Any) -> Any:
    """Convert a value into a JSON-comparable form (tuples -> lists)."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Stable JSON representation - the base of every argument digest."""
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))


def digest_of(value: Any) -> str:
    """Short SHA-256 digest of a canonical value (never the value itself)."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class ArgumentSpec:
    """One declared input of a tool."""

    name: str
    kind: str
    required: bool = True
    default: Any = None
    description: str = ""
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_name(self.name, "argument name")
        if self.kind not in _ARGUMENT_KINDS:
            raise ValueError(
                f"argument {self.name}: unknown kind {self.kind!r}; "
                f"allowed: {', '.join(sorted(_ARGUMENT_KINDS))}"
            )
        if self.max_length is not None and (not isinstance(self.max_length, int) or self.max_length < 1):
            raise ValueError(f"argument {self.name}: max_length must be a positive integer")
        if self.choices is not None:
            if not self.choices:
                raise ValueError(f"argument {self.name}: choices must not be empty")
            for choice in self.choices:
                self._check_type(choice, f"choice of {self.name}")
        if not self.required and self.default is not None:
            self._check_type(self.default, f"default of {self.name}")

    def _check_type(self, value: Any, what: str) -> None:
        expected = _ARGUMENT_KINDS[self.kind]
        if self.kind in {"integer", "number"}:
            if isinstance(value, bool) or not isinstance(value, expected):
                raise ValueError(f"{what} must be a number, got {type(value).__name__}")
        elif self.kind == "string":
            if not isinstance(value, str):
                raise ValueError(f"{what} must be a string, got {type(value).__name__}")
        elif self.kind == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{what} must be a boolean, got {type(value).__name__}")
        elif self.kind == "object":
            if not isinstance(value, dict):
                raise ValueError(f"{what} must be an object, got {type(value).__name__}")
        elif self.kind == "array":
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{what} must be an array, got {type(value).__name__}")

    def has_default(self) -> bool:
        return not self.required or self.default is not None

    def validate_value(self, value: Any) -> Any:
        """Validate one value and return it unchanged."""
        self._check_type(value, f"argument {self.name}")
        if self.kind in {"integer", "number"}:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"argument {self.name} must be a finite number")
            if self.minimum is not None and number < self.minimum:
                raise ValueError(f"argument {self.name} must be >= {self.minimum}")
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f"argument {self.name} must be <= {self.maximum}")
        if self.kind == "string" and self.max_length is not None and len(value) > self.max_length:
            raise ValueError(
                f"argument {self.name} must be at most {self.max_length} characters"
            )
        if self.choices is not None and not any(
            _canonical(value) == _canonical(choice) for choice in self.choices
        ):
            raise ValueError(f"argument {self.name} must be one of {list(self.choices)}")
        return value

    def describe(self) -> dict[str, Any]:
        """Provider-neutral description (usable by any AI layer)."""
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "description": self.description,
        }
        if self.choices is not None:
            payload["choices"] = list(self.choices)
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.max_length is not None:
            payload["max_length"] = self.max_length
        if not self.required or self.default is not None:
            payload["default"] = _canonical(self.default)
        return payload


@dataclass(frozen=True)
class ToolSpec:
    """Contract of a tool: identity, inputs and required permission."""

    name: str
    description: str
    version: str = "1.0.0"
    arguments: tuple[ArgumentSpec, ...] = ()
    permission: PermissionLevel = PermissionLevel.APPROVAL
    side_effects: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_name(self.name, "tool name")
        validate_description(self.description)
        validate_version(self.version)
        if len(self.arguments) > MAX_ARGUMENTS:
            raise ValueError(f"a tool may declare at most {MAX_ARGUMENTS} arguments")
        seen: set[str] = set()
        optional_seen = False
        for argument in self.arguments:
            if not isinstance(argument, ArgumentSpec):
                raise ValueError(f"tool {self.name}: arguments must be ArgumentSpec instances")
            if argument.name in seen:
                raise ValueError(f"tool {self.name}: duplicate argument {argument.name!r}")
            seen.add(argument.name)
            if not argument.required or argument.default is not None:
                optional_seen = True
            elif optional_seen:
                raise ValueError(
                    f"tool {self.name}: required argument {argument.name!r} "
                    "follows an optional one"
                )
        if self.permission is PermissionLevel.SAFE and self.side_effects:
            raise ValueError(
                f"tool {self.name}: a SAFE tool must declare side_effects=False"
            )
        if not isinstance(self.metadata, Mapping):
            raise ValueError(f"tool {self.name}: metadata must be a mapping")

    def argument(self, name: str) -> ArgumentSpec | None:
        for argument in self.arguments:
            if argument.name == name:
                return argument
        return None

    def validate_arguments(self, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        """Validate a raw argument mapping; unknown keys are rejected."""
        raw: Mapping[str, Any] = arguments or {}
        if not isinstance(raw, Mapping):
            raise ValueError("arguments must be a mapping")
        unknown = sorted(set(raw) - {argument.name for argument in self.arguments})
        if unknown:
            raise ValueError(
                f"tool {self.name}: unknown argument(s): {', '.join(unknown)}"
            )
        validated: dict[str, Any] = {}
        for argument in self.arguments:
            if argument.name in raw:
                validated[argument.name] = argument.validate_value(raw[argument.name])
            elif argument.default is not None:
                validated[argument.name] = argument.validate_value(argument.default)
            elif argument.required:
                raise ValueError(f"tool {self.name}: missing required argument {argument.name!r}")
        return validated

    def describe(self) -> dict[str, Any]:
        """Provider-neutral description, safe to hand to an AI layer."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "permission": self.permission.value,
            "side_effects": self.side_effects,
            "arguments": [argument.describe() for argument in self.arguments],
        }


@dataclass(frozen=True)
class ToolContext:
    """Everything a handler is allowed to know about its invocation."""

    execution_id: str
    caller: str = "unknown"
    deadline: float | None = None
    now: Callable[[], float] = time.time

    def remaining(self) -> float | None:
        """Seconds left, or None when no deadline was set."""
        if self.deadline is None:
            return None
        return self.deadline - self.now()

    def expired(self) -> bool:
        """Cooperative tools can check this and stop early."""
        remaining = self.remaining()
        return remaining is not None and remaining <= 0


ToolHandler = Callable[[Mapping[str, Any], ToolContext], Any]


@dataclass(frozen=True)
class Tool:
    """A validated specification plus its implementation."""

    spec: ToolSpec
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ToolSpec):
            raise ValueError("tool spec must be a ToolSpec")
        if not callable(self.handler):
            raise ValueError("tool handler must be callable")


@dataclass(frozen=True)
class ToolRequest:
    """Request to run one tool with concrete arguments."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    caller: str = "unknown"
    request_id: str = field(default_factory=lambda: new_id("req"))

    def __post_init__(self) -> None:
        validate_name(self.tool, "tool name")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("arguments must be a mapping")
        if len(self.arguments) > MAX_ARGUMENTS:
            raise ValueError(f"a request may carry at most {MAX_ARGUMENTS} arguments")

    def arguments_digest(self) -> str:
        """Digest of the canonical arguments - the anchor of an approval."""
        return digest_of(dict(self.arguments))

    def argument_keys(self) -> tuple[str, ...]:
        return tuple(sorted(str(key) for key in self.arguments))


@dataclass(frozen=True)
class ToolResult:
    """Structured outcome. `ok` is True only for a confirmed success."""

    execution_id: str
    tool: str
    status: ToolStatus
    output: Any = None
    error: str | None = None
    error_type: str | None = None
    permission: str = ""
    reason: str = ""
    duration_ms: float = 0.0
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.SUCCESS

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "tool": self.tool,
            "status": self.status.value,
            "ok": self.ok,
            "output": _canonical(self.output),
            "error": self.error,
            "error_type": self.error_type,
            "permission": self.permission,
            "reason": self.reason,
            "duration_ms": round(self.duration_ms, 3),
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a policy check performed right before execution."""

    level: PermissionLevel
    allowed: bool
    requires_approval: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "allowed": self.allowed,
            "requires_approval": self.requires_approval,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ApprovalToken:
    """Proof that a trusted caller agreed to one exact action.

    The token is bound to the tool name, its version and the digest of the
    arguments. Changing a significant argument invalidates it, so an
    approval can never be silently widened into a different action.
    """

    token_id: str
    tool: str
    version: str
    arguments_digest: str
    issued_at: float
    expires_at: float

    def matches(
        self,
        *,
        tool: str,
        version: str,
        digest: str,
        now: float,
    ) -> tuple[bool, str]:
        """Return (valid, reason). Never raises."""
        if tool != self.tool:
            return False, "approval was issued for a different tool"
        if version != self.version:
            return False, "tool version changed since approval"
        if digest != self.arguments_digest:
            return False, "arguments changed since approval"
        if now > self.expires_at:
            return False, "approval expired"
        return True, "valid"

    def describe(self) -> dict[str, Any]:
        """Secrets-free summary (no argument values)."""
        return {
            "token_id": self.token_id,
            "tool": self.tool,
            "version": self.version,
            "arguments_digest": self.arguments_digest,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class AuditRecord:
    """One execution attempt, safe to persist and to show to the user.

    Argument *values* are never stored: only the digest (to correlate with an
    approval) and the argument names (to explain what happened).
    """

    execution_id: str
    tool: str
    version: str
    timestamp: float
    status: str
    permission: str
    reason: str
    arguments_digest: str
    argument_keys: tuple[str, ...]
    duration_ms: float
    caller: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "tool": self.tool,
            "version": self.version,
            "timestamp": self.timestamp,
            "status": self.status,
            "permission": self.permission,
            "reason": self.reason,
            "arguments_digest": self.arguments_digest,
            "argument_keys": list(self.argument_keys),
            "duration_ms": round(self.duration_ms, 3),
            "caller": self.caller,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)


__all__ = [
    "MAX_ARGUMENTS",
    "MAX_DESCRIPTION_CHARS",
    "MAX_NAME_CHARS",
    "ArgumentSpec",
    "ApprovalToken",
    "AuditRecord",
    "PermissionDecision",
    "PermissionLevel",
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolRequest",
    "ToolResult",
    "ToolSpec",
    "ToolStatus",
    "canonical_json",
    "digest_of",
    "new_id",
    "validate_description",
    "validate_name",
    "validate_version",
]
