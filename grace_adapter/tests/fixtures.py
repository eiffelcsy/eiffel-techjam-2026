"""Tiny stand-ins so the core math is testable without any detector weights.

The one real detector left is a licence-gated Hub download, so a test that
needed its weights would not run in CI. Everything GRACE actually invents --
the adapter, the weighting, the losses, the fusion, the schedule, the cache --
is independent of which detector it is attached to, and is tested against
these. That independence is now only asserted by these fixtures: the second
real seam that used to demonstrate it went with the detector zoo.

`ToyDetector` mirrors the real arrangement rather than faking it: the detector
owns the parameters and `ToySplit` is a view over it, so `assert_frozen` and
`verify_split` exercise the same code paths they will on a real model.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from grace.splits.base import FeatureSpec, SplitDetector


class LinearHead(nn.Module):
    """The case where the input-Jacobian is a known constant, exactly `w`.

    Used to pin `head_gradient`: for a linear head the gradient must equal the
    weight row for every input, which is the property that lets one
    implementation cover linear and MLP heads with no branch.
    """

    def __init__(self, spec: FeatureSpec):
        super().__init__()
        self.w = nn.Parameter(torch.randn(spec.numel()) / spec.numel() ** 0.5)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return f.flatten(1) @ self.w


class MLPHead(nn.Module):
    """A nonlinear head, where E[h(f)] != h(E[f]) and the Jacobian varies."""

    def __init__(self, spec: FeatureSpec, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spec.numel(), hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f.flatten(1)).squeeze(-1)


class ToyPreprocess:
    """Module-level and picklable, unlike a lambda -- DataLoader forks this."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.resize((8, 8)).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


class ToyDetector(nn.Module):
    """Owns the weights, exactly as a real FrozenDetector does.

    Accepts any input shape and reads the first 8 values, so the same object
    serves both the flat-vector tests and the image-shaped probe `verify_split`
    builds.
    """

    name = "toy"

    def __init__(self, spec: FeatureSpec, head: nn.Module | None = None):
        super().__init__()
        self.spec = spec
        self.body = nn.Linear(8, spec.numel())
        self.head = head if head is not None else LinearHead(spec)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x.reshape(x.shape[0], -1)[:, :8]).reshape(x.shape[0], *self.spec.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def preprocess_fn(self):
        return ToyPreprocess()


class ToySplit(SplitDetector):
    """A SplitDetector satisfying head(trunk(x)) == detector(x) by construction.

    `n_taps > 0` also makes it emit taps, so the ladder -- cache views, reader,
    training loop, checkpoint round-trip -- is testable with no detector weights,
    exactly as everything else here is. The taps are a deterministic function of
    the seam features rather than genuine intermediate activations: what these
    tests check is the plumbing (shapes, row alignment, which view a tap came
    from), and a real trunk is not needed to get that wrong.
    """

    def __init__(
        self,
        spec: FeatureSpec,
        head: nn.Module | None = None,
        verify: bool = False,
        n_taps: int = 0,
    ):
        super().__init__(ToyDetector(spec, head))
        self._spec = spec
        self.n_taps = n_taps
        self.eval()
        self.requires_grad_(False)
        if verify:
            from grace.splits.verify import verify_split, verify_taps

            verify_split(self)
            verify_taps(self)

    @property
    def feature_spec(self) -> FeatureSpec:
        return self._spec

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.detector.trunk(x)

    def head(self, f: torch.Tensor) -> torch.Tensor:
        return self.detector.head(f)

    def taps(self) -> tuple[str, ...]:
        return tuple(f"block{k:02d}" for k in range(self.n_taps))

    def tap_spec(self) -> FeatureSpec | None:
        if not self.n_taps:
            return None
        return FeatureSpec(layout="layers", shape=(self.n_taps, self._spec.numel()))

    def trunk_with_taps(self, x: torch.Tensor):
        f = self.trunk(x)
        if not self.n_taps:
            return f, None
        # Distinct per tap and per image, so a test that crossed two taps or two
        # rows would see it. `f` itself is returned unchanged -- `verify_taps`
        # requires the tapped path to reproduce `trunk` exactly.
        flat = f.flatten(1)
        taps = torch.stack([flat * (k + 1) for k in range(self.n_taps)], dim=1)
        return f, taps


SPECS = {
    "vector": FeatureSpec(layout="vector", shape=(16,)),
    "tokens": FeatureSpec(layout="tokens", shape=(5, 16)),
    "layers": FeatureSpec(layout="layers", shape=(4, 16)),
}


def features(spec: FeatureSpec, batch: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, *spec.shape, generator=g)


def write_images(directory, n: int, seed: int = 0):
    """A manifest-shaped table over `n` generated PNGs."""
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        path = directory / f"{i:03d}.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
        rows.append({"path": str(path), "label": i % 2, "generator": "T", "split": "train"})
    return pd.DataFrame(rows)
