"""Splits are named by dotted import path, as detectors and datasets are.

    {target: "grace.splits.rine.RINESplit"}

No registry: a new detector is a new module here plus a config line, never an
edit to code in this package. `build_split` takes an already-constructed
detector, so a run loads its weights exactly once however many places need them.
"""

from grace.splits.base import LAYOUTS, FeatureSpec, SplitDetector
from pipeline.utils.imports import locate

__all__ = ["LAYOUTS", "FeatureSpec", "SplitDetector", "build_split"]


def build_split(detector, target: str, **kwargs) -> SplitDetector:
    """Wrap `detector` in the SplitDetector named by `target`."""
    cls = locate(target)
    split = cls(detector, **kwargs)
    if not isinstance(split, SplitDetector):
        raise TypeError(f"{target} produced {type(split).__name__}, not a SplitDetector")
    return split.eval()
