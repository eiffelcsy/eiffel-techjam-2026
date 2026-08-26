"""B-Free split -- the awkward one. Read this before caching it.

B-Free takes the image at *native resolution*, embeds the whole thing, cuts five
36x36 token windows in token space (center + four corners), scores each, and
averages the five logits. Two problems for a feature cache:

  1. The natural seam is per-window, so the trunk emits (B, 5, D), not (B, D).
     Cache all five and the head stays the untouched upstream average; average
     the five *features* instead and the split no longer reproduces `forward`.
     Take the first option -- correctness over 5x storage -- and let the adapter
     treat the window axis like tokens (shared weights, one gate).

  2. Input tensors are heterogeneously sized, which is why the harness pins
     `batch_size: 1` for B-Free. The cache writer inherits that: the *features*
     are fixed-size and batch fine, but the forward pass producing them does not.
     Caching is therefore slow for B-Free and fast for everything else -- and it
     is a one-time offline cost, which is precisely the argument for
     pre-rendering rather than degrading inside the training loop.

`pool="mean"` collapses the window axis to a vector layout at the cost of the
exact-equality check -- an ablation, not a default, and `verify_split` will
reject it, which is the correct behaviour.

VERIFY AGAINST THE CLONE -- see `grace.splits.verify`.
"""

import torch

from grace.splits.base import FeatureSpec, SplitDetector
from grace.splits.verify import verify_split

N_WINDOWS = 5
"""Center plus four corners, cut in token space by upstream's Wrapper5crops."""


class BFreeSplit(SplitDetector):
    """Cut B-Free at the five per-window embeddings, before the logit average."""

    def __init__(self, detector, pool: str | None = None, verify: bool = True):
        super().__init__(detector)
        self.pool = pool
        self._width = self._infer_width()
        if verify:
            verify_split(self)

    def _infer_width(self) -> int:
        """Embedding width of the DINOv2 ViT-B/14-with-registers backbone."""
        model = self.detector.model
        for attr in ("embed_dim", "num_features"):
            if hasattr(model, attr):
                return int(getattr(model, attr))
        raise RuntimeError(
            "could not infer B-Free's embedding width; set it explicitly once the "
            "clone is present (third_party/B-Free)."
        )

    @property
    def feature_spec(self) -> FeatureSpec:
        if self.pool == "mean":
            return FeatureSpec(layout="vector", shape=(self._width,))
        return FeatureSpec(layout="tokens", shape=(N_WINDOWS, self._width))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) at native resolution -> (B, 5, D) window embeddings."""
        raise NotImplementedError(
            "BFreeSplit.trunk needs third_party/B-Free: cut Wrapper5crops between "
            "its window extraction and its per-window classifier. See the module "
            "docstring; RINE is the development detector until this is done."
        )

    def head(self, f: torch.Tensor) -> torch.Tensor:
        """Per-window logits, averaged -- upstream's own aggregation, untouched."""
        raise NotImplementedError("see BFreeSplit.trunk")
