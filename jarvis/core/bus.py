"""In-process publish/subscribe event bus.

Design rules (kept deliberately small):

- No priorities, no middleware, no wildcard topics, no external broker.
- Handlers run in registration order, so behaviour is deterministic.
- A handler that raises never breaks the bus or starves the other handlers
  of the same event: the failure is recorded and logged, not swallowed.
- Recorded failures are bounded: at most `max_recorded_failures` are kept
  (oldest first out), so a long-lived process cannot grow without limit.
  Every failure is still logged, even when it is not retained.
- Thread safety: `subscribe`, `unsubscribe`, `clear`, `handlers` and failure
  recording are guarded by one lock, so a UI thread may subscribe while a
  worker thread publishes. Handlers themselves are *not* serialized: they are
  called outside the lock, so a slow handler never blocks registration, and a
  handler that mutates the bus still has to be quick and re-entrant safe.
- Synchronous delivery is `publish()`; `publish_async()` additionally awaits
  coroutine handlers. Registering an async handler and calling `publish()`
  is a recorded failure rather than a silent no-op.

Handlers must stay quick: enqueue heavy work instead of doing it inline.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

APP_STARTING = "app.starting"
APP_RUNNING = "app.running"
APP_STOPPING = "app.stopping"
APP_STOPPED = "app.stopped"


@dataclass(frozen=True)
class Event:
    """A single bus message: a type plus an immutable-by-convention payload."""

    type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


@dataclass(frozen=True)
class BusFailure:
    """Recorded handler failure - useful for diagnostics and tests."""

    event_type: str
    handler: str
    error: BaseException


Handler = Callable[[Event], Any]


DEFAULT_MAX_RECORDED_FAILURES = 50


class EventBus:
    """Minimal deterministic event bus."""

    def __init__(
        self,
        *,
        record_failures: bool = True,
        max_recorded_failures: int = DEFAULT_MAX_RECORDED_FAILURES,
    ) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self.failures: list[BusFailure] = []
        self._record_failures = bool(record_failures)
        self._max_recorded_failures = int(max_recorded_failures)
        # Guards registration and the failure log; handlers run outside it.
        self._lock = threading.RLock()

    def subscribe(self, event_type: str, handler: Handler) -> Callable[[], None]:
        """Register a handler and return an unsubscribe callable."""
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

        def _unsubscribe() -> None:
            self.unsubscribe(event_type, handler)

        return _unsubscribe

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)
                if not handlers:
                    del self._handlers[event_type]

    def handlers(self, event_type: str) -> tuple[Handler, ...]:
        """Return handlers for a type in registration order."""
        with self._lock:
            return tuple(self._handlers.get(event_type, ()))

    def clear(self, event_type: str | None = None) -> None:
        """Drop handlers: one type or the whole bus."""
        with self._lock:
            if event_type is None:
                self._handlers.clear()
            else:
                self._handlers.pop(event_type, None)

    def clear_failures(self) -> None:
        with self._lock:
            self.failures.clear()

    def publish(self, event: Event) -> list[Any]:
        """Deliver an event synchronously; returns per-handler results."""
        results: list[Any] = []
        for handler in self.handlers(event.type):
            if inspect.iscoroutinefunction(handler):
                self._record(
                    event,
                    handler,
                    TypeError(
                        f"async handler {handler!r} registered for "
                        f"{event.type!r}: use publish_async() instead of publish()"
                    ),
                )
                results.append(None)
                continue
            try:
                results.append(handler(event))
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                self._record(event, handler, exc)
                results.append(None)
        return results

    async def publish_async(self, event: Event) -> list[Any]:
        """Deliver an event, awaiting coroutine handlers in order."""
        results: list[Any] = []
        for handler in self.handlers(event.type):
            try:
                value = handler(event)
                if inspect.isawaitable(value):
                    value = await value
                results.append(value)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                self._record(event, handler, exc)
                results.append(None)
        return results

    def _record(self, event: Event, handler: Handler, error: BaseException) -> None:
        logger.error(
            "event bus handler failed (event=%s, handler=%r)",
            event.type,
            handler,
            exc_info=error,
        )
        if not self._record_failures or self._max_recorded_failures <= 0:
            return
        with self._lock:
            self.failures.append(
                BusFailure(event_type=event.type, handler=repr(handler), error=error)
            )
            while len(self.failures) > self._max_recorded_failures:
                del self.failures[0]
