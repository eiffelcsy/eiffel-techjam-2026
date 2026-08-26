"""Make `grace`, `pipeline` and `tests` importable without an editable install.

Mirrors eval_pipeline/conftest.py, plus the sibling harness: GRACE is a layer on
top of it, not a fork of it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval_pipeline"))
