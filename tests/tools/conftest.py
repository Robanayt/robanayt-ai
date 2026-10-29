"""Fixtures for tool-system tests. Shared helpers live in `tests.tools.helpers`."""

from __future__ import annotations

import pytest

from tests.tools.helpers import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
