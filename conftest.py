"""Make the repo-root modules importable from tests/ without editing them.

The modules live flat at the repo root (intent_monitor.py, fixtures.py,
model.py, evaluate.py, demo.py) and import each other flat. Placing this
conftest at the repo root puts the root on sys.path for the whole test session,
so `from fixtures import ...` etc. resolve when tests run from `tests/`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
