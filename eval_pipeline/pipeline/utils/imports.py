"""Import-path component resolution.

Detectors and dataset sources are named in config by dotted import path, not by
a registry key. That keeps the pipeline agnostic: adding a component is a new
module plus a config line, never an edit to code in here.

    {target: "pipeline.detectors.dinov3.DINOv3MLPDetector", args: {backbone_id: ...}}
"""

import importlib
import sys
from pathlib import Path
from typing import Any


def _ensure_cwd_importable() -> None:
    """Put the working directory on sys.path.

    `python scripts/run_eval.py` puts `scripts/` on sys.path, not the directory
    the user is standing in -- so without this, a detector or source they wrote
    in their own project could be named by import path but never imported, and
    only installed packages would resolve. Same reason uvicorn and hydra do it.
    """
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def locate(path: str) -> Any:
    """Resolve a dotted path like "pkg.mod.Class" to the object itself."""
    if "." not in path:
        raise ValueError(f"not a dotted import path: {path!r}")
    _ensure_cwd_importable()
    module_path, _, attr = path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"cannot import module {module_path!r} for target {path!r}") from e
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"module {module_path!r} has no attribute {attr!r}") from e


def instantiate(spec: dict, **overrides: Any) -> Any:
    """Build the object named by spec["target"], called with spec["args"].

    Keys other than `target` and `args` are ignored, so a spec can carry
    bookkeeping fields (`name`, `device`) alongside constructor arguments.
    """
    if "target" not in spec:
        raise KeyError(f"spec is missing required key 'target': {spec!r}")
    factory = locate(spec["target"])
    args = {**(spec.get("args") or {}), **overrides}
    return factory(**args)
