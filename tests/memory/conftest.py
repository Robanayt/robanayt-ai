"""Memory tests import shared fakes by package path (`tests.ai.fakes`).

Kept empty on purpose: patching `sys.path` here made the bare module names
`fakes` and `conftest` ambiguous for other suites collected in the same
session.
"""
