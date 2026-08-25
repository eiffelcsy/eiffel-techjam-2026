"""Detector construction.

Detectors are named in config by dotted import path, so this package holds no
registry of known models -- only the contract (`FrozenDetector`), one generic
Hub adapter (`hf.HFImageClassifier`), and the two functions that turn a config
entry into a loaded model on the right device.
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
