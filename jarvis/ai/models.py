"""Provider-agnostic AI data models.

These structures are deliberately vendor-neutral: no Ollama field names, no
OpenAI field names. Providers translate to and from their own wire format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from jarvis.ai.errors import AIRequestError

SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
TOOL = "tool"

VALID_ROLES: frozenset[str] = frozenset({SYSTEM, USER, ASSISTANT, TOOL})


@dataclass(frozen=True)
class AIMessage:
    """A single role-tagged message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise AIRequestError(
                f"invalid message role {self.role!r}; allowed: {sorted(VALID_ROLES)}"
            )
        if not isinstance(self.content, str):
            raise AIRequestError("message content must be a string")

    @classmethod
    def system(cls, content: str) -> AIMessage:
        return cls(role=SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> AIMessage:
        return cls(role=USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> AIMessage:
        return cls(role=ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str) -> AIMessage:
        return cls(role=TOOL, content=content)

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Usage:
    """Token usage as reported by a provider (all fields optional)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class AITool:
    """Provider-neutral description of one callable tool.

    The AI layer never imports the tool system: the orchestrator translates
    registered tool specifications into this shape. `parameters` is a
    JSON-Schema-compatible object; it is forwarded to the provider verbatim
    and is never treated as instructions. Declaring a tool never grants the
    right to call it - that decision stays in `jarvis.tools.policy`.
    """

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AIRequestError("a tool declaration requires a non-empty name")
        if not isinstance(self.description, str):
            raise AIRequestError("a tool description must be a string")
        if not isinstance(self.parameters, Mapping):
            raise AIRequestError("tool parameters must be a mapping")
        object.__setattr__(self, "name", self.name.strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class AIRequest:
    """A generation request. Never carries credentials."""

    messages: tuple[AIMessage, ...] = ()
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[AITool, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.messages, AIMessage):  # defensive: single message
            object.__setattr__(self, "messages", (self.messages,))
        else:
            object.__setattr__(self, "messages", tuple(self.messages))
        if self.tools is None:
            object.__setattr__(self, "tools", ())
        elif isinstance(self.tools, AITool):  # defensive: single declaration
            object.__setattr__(self, "tools", (self.tools,))
        else:
            object.__setattr__(self, "tools", tuple(self.tools))

    def validate(self) -> None:
        """Raise `AIRequestError` when the request cannot be sent."""
        if not self.messages:
            raise AIRequestError("request has no messages")
        for message in self.messages:
            if not isinstance(message, AIMessage):
                raise AIRequestError(f"invalid message object: {message!r}")
        seen: set[str] = set()
        for tool in self.tools:
            if not isinstance(tool, AITool):
                raise AIRequestError(f"invalid tool declaration: {tool!r}")
            if tool.name in seen:
                raise AIRequestError(f"duplicate tool declaration {tool.name!r}")
            seen.add(tool.name)
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise AIRequestError(f"temperature out of range: {self.temperature}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise AIRequestError(f"max_tokens must be positive: {self.max_tokens}")
        if self.timeout is not None and self.timeout <= 0:
            raise AIRequestError(f"timeout must be positive: {self.timeout}")

    def with_overrides(
        self,
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> AIRequest:
        """Return a copy with provider-level defaults applied."""
        updates: dict[str, Any] = {}
        if model and not self.model:
            updates["model"] = model
        if timeout is not None and self.timeout is None:
            updates["timeout"] = timeout
        return replace(self, **updates) if updates else self

    def requested_provider(self) -> str | None:
        """Provider name pinned through metadata, if any."""
        value = self.metadata.get("provider") if self.metadata else None
        return str(value) if value else None


@dataclass(frozen=True)
class AIResponse:
    """Normalised generation result."""

    content: str
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content


@dataclass(frozen=True)
class AIChunk:
    """One incremental piece of a streamed response."""

    delta: str = ""
    done: bool = False
    finish_reason: str | None = None
    model: str | None = None
    provider: str | None = None

    @classmethod
    def final(
        cls,
        *,
        model: str | None = None,
        provider: str | None = None,
        finish_reason: str = "stop",
    ) -> AIChunk:
        return cls(
            delta="",
            done=True,
            finish_reason=finish_reason,
            model=model,
            provider=provider,
        )


@dataclass(frozen=True)
class ProviderStatus:
    """Configuration/availability state of one provider."""

    provider: str
    configured: bool = False
    available: bool = False
    message: str = ""
    checked_at: float | None = None

    @property
    def ok(self) -> bool:
        return self.configured and self.available

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "configured": self.configured,
            "available": self.available,
            "message": self.message,
        }


def messages_from_pairs(pairs: Sequence[tuple[str, str]]) -> tuple[AIMessage, ...]:
    """Build messages from `(role, content)` pairs."""
    return tuple(AIMessage(role=role, content=content) for role, content in pairs)
