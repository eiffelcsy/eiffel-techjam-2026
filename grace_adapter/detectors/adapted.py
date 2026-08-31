import torch
from PIL import Image

from grace_adapter.models.factory import load_adapter
from grace_adapter.models.severity import SeverityHead
from eval.splits import build_split
from eval.config import load_detector_config
from eval.detectors import build_detector
from eval.detectors.base import FrozenDetector

class AdaptedDetector(FrozenDetector):
    """A frozen detector with a trained GRACE adapter spliced at its seam.

    Parameters
    ----------
    base        : path to the base detector's config (or the same mapping inline)
    split       : dotted relative path to the SplitDetector for that base
    checkpoint  : trained adapter weights; None = identity, i.e. exactly the base
                  detector.
    """

    def __init__(
        self,
        base,
        split: str,
        checkpoint: str | None = None,
        name: str = "grace",
        split_args: dict | None = None,
    ):
        super().__init__()
        detector = build_detector(load_detector_config(base))
        self.split = build_split(detector, split, **(split_args or {}))
        self.name = name

        spec = self.split.feature_spec
        self.adapter = load_adapter(checkpoint, spec) if checkpoint else None
        self.severity_head = None

        if self.adapter is not None and self.adapter.film is not None:
            # The severity head ships inside the stage-1 checkpoint; without it
            # the FiLM path has no input.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "severity_state_dict" in payload:
                self.severity_head = SeverityHead(spec.dim)
                self.severity_head.load_state_dict(payload["severity_state_dict"])

        self.freeze()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.split.detector.preprocess(img)

    def preprocess_fn(self):
        return self.split.preprocess_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.split.trunk(x)
        if self.adapter is None:
            return self.split.head(f)

        f = f.float()
        severity = self.severity_head(f) if self.severity_head is not None else None

        f_adapted = self.adapter(f, severity=severity)
        return self.split.head(f_adapted)
