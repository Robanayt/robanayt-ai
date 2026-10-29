"""Event bus behaviour: ordering, isolation, sync/async delivery, concurrency."""

from __future__ import annotations

import asyncio
import threading

from robanayt_ai.core.bus import Event, EventBus


def test_handlers_run_in_registration_order() -> None:
    bus = EventBus()
    calls: list[str] = []
    bus.subscribe("x", lambda event: calls.append(f"a:{event.type}"))
    bus.subscribe("x", lambda event: calls.append(f"b:{event.type}"))

    bus.publish(Event("x", {"n": 1}))

    assert calls == ["a:x", "b:x"]


def test_concurrent_publish_is_safe() -> None:
    """The UI thread subscribes while a worker thread publishes."""
    bus = EventBus()
    hits: list[int] = []
    lock = threading.Lock()

    def handler(event: Event) -> None:
        with lock:
            hits.append(int(event.get("n", 0)))

    bus.subscribe("x", handler)

    def worker(base: int) -> None:
        for index in range(50):
            bus.publish(Event("x", {"n": base * 100 + index}))

    threads = [threading.Thread(target=worker, args=(base,)) for base in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(hits) == 400
    assert len(set(hits)) == 400


def test_subscribe_during_publish_is_safe() -> None:
    bus = EventBus()
    bus.subscribe("x", lambda event: None)

    def subscriber() -> None:
        for _ in range(50):
            bus.subscribe("y", lambda event: None)

    def publisher() -> None:
        for _ in range(50):
            bus.publish(Event("x", {"n": 1}))

    threads = [
        threading.Thread(target=subscriber),
        threading.Thread(target=publisher),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(bus.handlers("y")) == 50
    assert bus.failures == []


def test_publish_returns_handler_results() -> None:
    bus = EventBus()
    bus.subscribe("x", lambda event: event.get("n", 0) * 2)

    assert bus.publish(Event("x", {"n": 21})) == [42]


def test_unknown_event_type_is_a_noop() -> None:
    bus = EventBus()

    assert bus.publish(Event("nobody")) == []
    assert bus.failures == []


def test_failing_handler_does_not_stop_the_others() -> None:
    bus = EventBus()
    calls: list[str] = []

    def boom(_event: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe("x", boom)
    bus.subscribe("x", lambda event: calls.append(event.type))

    results = bus.publish(Event("x"))

    assert calls == ["x"]
    assert results == [None, None]
    assert len(bus.failures) == 1
    assert isinstance(bus.failures[0].error, RuntimeError)


def test_async_handler_with_sync_publish_is_recorded_not_silent() -> None:
    bus = EventBus()

    async def handler(_event: Event) -> str:
        return "never"

    bus.subscribe("x", handler)

    assert bus.publish(Event("x")) == [None]
    assert isinstance(bus.failures[0].error, TypeError)


def test_publish_async_awaits_coroutine_handlers() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: Event) -> str:
        seen.append(event.type)
        return "ok"

    async def sync_handler(event: Event) -> str:
        seen.append(f"sync:{event.type}")
        return "sync"

    bus.subscribe("x", handler)
    bus.subscribe("x", sync_handler)

    assert asyncio.run(bus.publish_async(Event("x"))) == ["ok", "sync"]
    assert seen == ["x", "sync:x"]
    assert bus.failures == []


def test_publish_async_records_failures() -> None:
    bus = EventBus()

    async def handler(_event: Event) -> None:
        raise ValueError("nope")

    bus.subscribe("x", handler)

    assert asyncio.run(bus.publish_async(Event("x"))) == [None]
    assert isinstance(bus.failures[0].error, ValueError)


def test_unsubscribe_and_clear() -> None:
    bus = EventBus()
    calls: list[str] = []

    def handler(_event: Event) -> None:
        calls.append("hit")

    unsubscribe = bus.subscribe("x", handler)
    bus.publish(Event("x"))
    assert calls == ["hit"]

    unsubscribe()
    assert bus.handlers("x") == ()
    bus.publish(Event("x"))
    assert calls == ["hit"]

    bus.subscribe("y", handler)
    bus.clear()
    assert bus.handlers("y") == ()


def test_recorded_failures_are_bounded() -> None:
    bus = EventBus(max_recorded_failures=3)

    def boom(_event: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe("x", boom)
    for _ in range(10):
        bus.publish(Event("x"))

    assert len(bus.failures) == 3


def test_failure_recording_can_be_disabled() -> None:
    def boom(_event: Event) -> None:
        raise RuntimeError("boom")

    quiet = EventBus(record_failures=False)
    quiet.subscribe("x", boom)
    quiet.publish(Event("x"))

    capped = EventBus(max_recorded_failures=0)
    capped.subscribe("x", boom)
    capped.publish(Event("x"))

    assert quiet.failures == []
    assert capped.failures == []
