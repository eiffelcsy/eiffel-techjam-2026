"""RINE split -- the development detector, because its seam already exists.

RINE *is* a trunk/head split: CLIP ViT-L/14 stays frozen while forward hooks pull
the CLS token out of all 24 transformer blocks, and everything trainable lives in
a small head (their Trainable Importance Estimator, a projection, an MLP). So:

    trunk = frozen CLIP + hook collection   -> (B, 24, 1024), layout "layers"
    head  = the trained head                -> (B,) logit

Two consequences:

  * The `layers` layout is the interesting case. A per-layer gate lets GRACE
    learn *which blocks a given degradation destroys* -- blur should wreck the
    early high-frequency blocks and leave late semantic ones intact. That gate
    vector is a figure, not just a parameter.
  * It is also the expensive case to cache: 24x1024 float16 = 48 KB per image per
    view. See the cost table in README.md.

`tencrop=True` multiplies the trunk output by ten and is not cacheable at that
size, so it is rejected here.

--------------------------------------------------------------------------------
VERIFY AGAINST THE CLONE. `_head_forward` reproduces upstream's composition of
the trained modules from `third_party/rine`, which is not vendored in this repo.
`__init__` therefore runs `verify_split()` -- one random batch through both paths
-- and raises if `head(trunk(x))` does not reproduce `detector(x)`. A wrong
composition is then an immediate, actionable error rather than a silently
different model that invalidates every comparison downstream.
--------------------------------------------------------------------------------
"""

import torch

from grace.splits.base import FeatureSpec, SplitDetector
from grace.splits.verify import verify_split

CLIP_CHILD = "clip"
"""RINE's checkpoints strip CLIP's keys at save time (`FROZEN_PREFIX = "clip."`
in the detector adapter), so *every child that is not this one* is trained head.
That is the rule `head_modules` relies on rather than a hardcoded list."""


class RINESplit(SplitDetector):
    """Cut RINE at the stacked per-block CLS tokens."""

    def __init__(self, detector, verify: bool = True):
        super().__init__(detector)
        if getattr(detector, "tencrop", False):
            raise ValueError(
                "RINESplit does not support tencrop=True: the trunk would emit "
                "10x the features and the cache is already the largest in the "
                "project. Use the center-crop protocol for adapter work."
            )
        self._blocks = self._resblocks()
        self._width = self.detector.model.clip.visual.transformer.width
        if verify:
            verify_split(self)

    def _resblocks(self):
        """The 24 residual blocks whose CLS tokens are the features."""
        return self.detector.model.clip.visual.transformer.resblocks

    @property
    def feature_spec(self) -> FeatureSpec:
        # Read the depth and width off the model: a ViT-B checkpoint would work
        # unchanged, and a hardcoded (24, 1024) would not.
        return FeatureSpec(layout="layers", shape=(len(self._blocks), self._width))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, L, D): the CLS token from every encoder block."""
        collected: list[torch.Tensor] = []

        def hook(_module, _inp, out):
            # CLIP's resblocks run sequence-first (T, B, D); the CLS token is
            # position 0. Upstream takes the same slice.
            collected.append(out[0] if out.ndim == 3 else out)

        handles = [b.register_forward_hook(hook) for b in self._blocks]
        try:
            self.detector.model.clip.encode_image(x)
        finally:
            for h in handles:
                h.remove()
        return torch.stack(collected, dim=1)

    def head(self, f: torch.Tensor) -> torch.Tensor:
        return self._head_forward(f)

    def _head_forward(self, f: torch.Tensor) -> torch.Tensor:
        """The trained head. VERIFY AGAINST THE CLONE -- see the module docstring.

        Upstream shape: project each block's CLS token, weight the blocks by the
        softmax-normalised importance estimator, sum, then run the classifier MLP
        to one logit.
        """
        model = self.detector.model
        return model.head(model.proj(f), model.alpha) if hasattr(model, "alpha") else None

    def head_modules(self) -> dict:
        """Everything trainable, i.e. everything that is not CLIP.

        Printed by `verify_split`'s failure message so that fixing
        `_head_forward` does not require reading the clone from scratch.
        """
        return {n: m for n, m in self.detector.model.named_children() if n != CLIP_CHILD}
