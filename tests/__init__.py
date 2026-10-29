"""Test package marker.

pytest's default "prepend" import mode imports a test file under its module
name. Without package markers every file would be imported top-level, so two
files with the same basename in different directories (for example
`tests/ai/test_models.py` and `tests/memory/test_models.py`) would collide
with "import file mismatch".

Keeping the test tree as real packages gives each file a unique dotted module
name (`tests.memory.test_models`) while all production code stays untouched.
"""
