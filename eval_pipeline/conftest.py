"""Make `pipeline` and `tests` importable without an editable install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
