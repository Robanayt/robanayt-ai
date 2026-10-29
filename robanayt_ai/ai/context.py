"""Provider-independent conversation context.

This is the abstraction future memory will plug into - not the memory itself:
no database, no embeddings, no vector search, no long-term storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from robanayt_ai.ai.models import AIRequest, AIMessage

DEFAULT_MAX_MESSAGES = 20


class ConversationContext:
    """System prompt plus a bounded rolling history."""

    def __init__(
        self,
        system_prompt: str | None = None,
        *,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        messages: Iterable[AIMessage] | None = None,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        self._system_prompt = system_prompt
        self._max_messages = max_messages
        self._messages: list[AIMessage] = list(messages or ())

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def max_messages(self) -> int:
        return self._max_messages

    @property
    def messages(self) -> tuple[AIMessage, ...]:
        """Conversation messages without the system prompt."""
        return tuple(self._messages)

    def set_system(self, content: str | None) -> None:
        self._system_prompt = content

    def add(self, message: AIMessage) -> AIMessage:
        """Append a message and enforce the history limit."""
        if not isinstance(message, AIMessage):
            raise TypeError("add() expects an AIMessage")
        self._messages.append(message)
        self._trim()
        return message

    def add_user(self, content: str) -> AIMessage:
        return self.add(AIMessage.user(content))

    def add_assistant(self, content: str) -> AIMessage:
        return self.add(AIMessage.assistant(content))

    def extend(self, messages: Sequence[AIMessage]) -> None:
        for message in messages:
            self.add(message)

    def clear(self) -> None:
        self._messages.clear()

    def history(self, limit: int | None = None) -> tuple[AIMessage, ...]:
        """Most recent messages, oldest first."""
        if limit is None:
            return self.messages
        if limit <= 0:
            return ()
        return tuple(self._messages[-limit:])

    def to_messages(self) -> tuple[AIMessage, ...]:
        """Request-ready messages: system prompt first, then trimmed history."""
        messages: list[AIMessage] = []
        if self._system_prompt:
            messages.append(AIMessage.system(self._system_prompt))
        messages.extend(self.history())
        return tuple(messages)

    def to_request(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AIRequest:
        """Build a validated `AIRequest` from the current context."""
        return AIRequest(
            messages=self.to_messages(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            metadata=dict(metadata or {}),
        )

    def __len__(self) -> int:
        return len(self._messages)

    def _trim(self) -> None:
        if len(self._messages) > self._max_messages:
            del self._messages[: len(self._messages) - self._max_messages]
