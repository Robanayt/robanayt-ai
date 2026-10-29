"""Permission policy: safe, approval-bound and denied execution."""

from __future__ import annotations

import pytest

from tests.tools.helpers import FakeClock, approval_spec, denied_spec, safe_spec
from jarvis.tools.models import PermissionLevel, digest_of
from jarvis.tools.policy import PermissionPolicy


def test_safe_tool_is_allowed() -> None:
    policy = PermissionPolicy(clock=FakeClock())
    decision = policy.evaluate(safe_spec())

    assert decision.allowed is True
    assert decision.requires_approval is False


def test_unknown_tool_is_denied_by_default() -> None:
    policy = PermissionPolicy(clock=FakeClock())
    decision = policy.evaluate(approval_spec())

    assert decision.allowed is False
    assert decision.level is PermissionLevel.DENIED
    assert "unknown" in decision.reason


def test_declared_denied_tool_is_denied() -> None:
    policy = PermissionPolicy(clock=FakeClock())
    decision = policy.evaluate(denied_spec())

    assert decision.allowed is False
    assert "forbidden" in decision.reason


def test_policy_entry_overrides_declaration() -> None:
    policy = PermissionPolicy({"ping": PermissionLevel.DENIED}, clock=FakeClock())
    assert policy.evaluate(safe_spec("ping")).allowed is False

    approval_policy = PermissionPolicy({"touch": PermissionLevel.APPROVAL}, clock=FakeClock())
    decision = approval_policy.evaluate(approval_spec("touch"))
    assert decision.requires_approval is True
    assert decision.allowed is False


def test_allow_declared_safe_can_be_disabled() -> None:
    policy = PermissionPolicy(allow_declared_safe=False, clock=FakeClock())
    assert policy.evaluate(safe_spec()).allowed is False


def test_grant_and_verify() -> None:
    clock = FakeClock()
    policy = PermissionPolicy(clock=clock)
    spec = approval_spec()
    token = policy.grant(spec, digest_of({"a": 1}))

    valid, reason = policy.verify(
        token, tool=spec.name, version=spec.version, digest=digest_of({"a": 1})
    )
    assert valid is True and reason == "valid"


def test_verify_without_token_fails() -> None:
    policy = PermissionPolicy(clock=FakeClock())
    valid, reason = policy.verify(None, tool="touch", version="1.0.0", digest=digest_of({}))
    assert valid is False and "no approval" in reason


def test_verify_rejects_changed_arguments() -> None:
    policy = PermissionPolicy(clock=FakeClock())
    spec = approval_spec()
    token = policy.grant(spec, digest_of({"path": "a.txt"}))

    valid, reason = policy.verify(
        token, tool=spec.name, version=spec.version, digest=digest_of({"path": "b.txt"})
    )
    assert valid is False and "arguments changed" in reason


def test_verify_rejects_expired_token() -> None:
    clock = FakeClock()
    policy = PermissionPolicy(approval_ttl_seconds=10, clock=clock)
    spec = approval_spec()
    token = policy.grant(spec, digest_of({}))

    clock.advance(11)
    valid, reason = policy.verify(
        token, tool=spec.name, version=spec.version, digest=digest_of({})
    )
    assert valid is False and "expired" in reason


def test_token_is_single_use() -> None:
    policy = PermissionPolicy(clock=FakeClock())
    spec = approval_spec()
    token = policy.grant(spec, digest_of({}))
    key = {"tool": spec.name, "version": spec.version, "digest": digest_of({})}

    assert policy.verify(token, **key)[0] is True
    policy.consume(token)
    assert policy.verify(token, **key)[0] is False


def test_revoke_prevents_use() -> None:
    policy = PermissionPolicy(clock=FakeClock())
    spec = approval_spec()
    token = policy.grant(spec, digest_of({}))
    policy.revoke(token)

    valid, reason = policy.verify(
        token, tool=spec.name, version=spec.version, digest=digest_of({})
    )
    assert valid is False and "already been used" in reason


def test_consumed_approvals_are_bounded() -> None:
    policy = PermissionPolicy(clock=FakeClock(), max_consumed=2)
    tokens = [policy.grant("custom", digest_of({"n": index})) for index in range(5)]

    for token in tokens:
        policy.consume(token)

    assert len(policy._consumed) == 2


def test_newest_approval_stays_consumed_after_eviction() -> None:
    policy = PermissionPolicy(clock=FakeClock(), max_consumed=2)
    tokens = [policy.grant("custom", digest_of({"n": index})) for index in range(5)]
    for token in tokens:
        policy.consume(token)

    newest = tokens[-1]
    valid, reason = policy.verify(
        newest, tool="custom", version="1.0.0", digest=digest_of({"n": 4})
    )
    assert valid is False and "already been used" in reason


def test_grant_accepts_a_plain_name() -> None:
    policy = PermissionPolicy(clock=FakeClock())
    token = policy.grant("custom", digest_of({}), version="3.1.4")
    assert token.tool == "custom" and token.version == "3.1.4"


def test_describe_reports_policy_shape() -> None:
    policy = PermissionPolicy({"ping": PermissionLevel.SAFE}, clock=FakeClock())
    description = policy.describe()

    assert description["known"] == {"ping": "safe"}
    assert description["deny_unknown_by_default"] is True
