"""Detector construction.

Detectors are named in config by dotted import path, so this package holds no
registry of known models -- only the contract (`FrozenDetector`), the adapters,
and the two functions that turn a config entry into a loaded model on the right
device.

`hf.HFImageClassifier` covers any Hub `AutoModelForImageClassification` from
config alone. `bfree`, `gapl` and `rine` are the zoo: published detectors whose
repos are not pip-installable, cloned by hand under `third_party/` and imported
through `_vendor.vendored`. Each defers its upstream imports into `__init__`, so
this package stays importable with none of them present.
"""

import torch

from pipeline.detectors.base import FrozenDetector
from pipeline.utils.imports import instantiate

__all__ = ["FrozenDetector", "build_detector", "resolve_device"]


def resolve_device(device: str = "auto") -> torch.device:
    """"auto" picks the best available accelerator; anything else is taken as given.

    Keeps one config runnable on a GPU cluster node, a Mac, and CPU-only CI.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_detector(cfg) -> FrozenDetector:
    """Instantiate a DetectorConfig and move it to its device, frozen."""
    detector = instantiate({"target": cfg.target, "args": cfg.args})
    if not isinstance(detector, FrozenDetector):
        raise TypeError(
            f"{cfg.target} produced {type(detector).__name__}, "
            "which is not a FrozenDetector"
        )
    detector.name = cfg.name or detector.name
    return detector.to(resolve_device(cfg.device)).freeze()
