"""A dataset and a detector that need no network, no weights, and no Hub.

The pair is built so the expected clean AUC is analytic rather than eyeballed:
`SyntheticSource` plants a label signal in the red channel with a margin far
wider than the per-image sampling noise, and `StubDetector` reads exactly that
signal. Clean AUC is therefore 1.0 by construction, and any deviation is a bug
in the harness rather than a property of a model.

The planted signal is deliberately fragile in a realistic way -- brightness
jitter, blur and noise all disturb a channel mean -- so the degraded levels
produce non-trivial retention and the composed levels genuinely exercise the
aggregation path.

Image sizes vary deliberately, including non-square and small, because the
harness promises one grid applies unchanged to a mixed-resolution dataset.
"""

import numpy as np
import torch
from PIL import Image

from pipeline.detectors.base import FrozenDetector

SIZES = [(32, 32), (48, 32), (64, 64), (127, 89), (256, 192), (512, 384)]
RED_OFFSET = {0: 40, 1: 80}     # real, generated -- a 40/255 margin
FLOOR, SPREAD = 60, 30


class SyntheticSource:
    """Random images carrying a planted, perfectly separable label signal."""

    def __init__(self, n_per_class: int = 10, seed: int = 0, split: str = "val",
                 sizes=SIZES, generator: str = "synthetic-v1"):
        self.n_per_class = n_per_class
        self.seed = seed
        self.split = split
        self.sizes = sizes
        self.generator = generator

    def rows(self, out_dir):
        from pathlib import Path

        rng = np.random.default_rng(self.seed)
        out_dir = Path(out_dir)
        for label in (0, 1):
            for i in range(self.n_per_class):
                w, h = self.sizes[i % len(self.sizes)]
                arr = rng.integers(0, SPREAD, (h, w, 3), dtype=np.int16) + FLOOR
                arr[..., 0] += RED_OFFSET[label]
                img = Image.fromarray(arr.astype(np.uint8), mode="RGB")

                path = out_dir / "images" / f"{label}_{i:04d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path, "PNG")
                yield {
                    "path": str(path.resolve()),
                    "label": label,
                    "generator": self.generator if label else "REAL",
                    "split": self.split,
                }


class StubDetector(FrozenDetector):
    """Reads the planted signal: mean red channel -> a fake-vs-real logit.

    Mirrors a real detector's shape -- it resizes to a fixed input in
    `preprocess`, so varied source resolutions must survive the transform chain
    to be batchable at all.
    """

    def __init__(self, input_size: int = 16, scale: float = 20.0, name: str = "stub"):
        super().__init__()
        self.input_size = input_size
        self.scale = scale
        self.name = name
        # Midpoint of the two planted red means, in [0,1] units.
        self.midpoint = (FLOOR + SPREAD / 2 + np.mean(list(RED_OFFSET.values()))) / 255.0
        self.freeze()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        small = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, 0].mean(dim=(1, 2)) - self.midpoint) * self.scale
