"""Make `tests.fixtures` resolve to grace_adapter/tests/fixtures.py.

grace_adapter/tests/ and the repo-root tests/ are both real packages named
`tests` (each has an __init__.py), which pytest's default "prepend" import
mode cannot disambiguate by sys.path order alone -- root pyproject.toml sets
`--import-mode=importlib` for exactly this reason; see its comment. This
insert is still needed for the plain `from tests.fixtures import ...` calls
inside grace_adapter's own test modules to find the right one. Retired once
grace_adapter/tests merges into the root tests/ tree and the name collision
goes away with it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

