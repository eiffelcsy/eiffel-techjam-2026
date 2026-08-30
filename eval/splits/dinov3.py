"""DINOv3 split -- the proof-of-concept seam, and the only one that runs today.

The only one at all, now that the zoo splits are gone. They each reconstructed
a seam inside a vendored repo that was never in this tree, and RINE's head
composition carried a standing warning that it must be checked against its
clone. That is the right design for adapting somebody else's published
detector, and it was a bad way to find out whether GRACE works at all: a
retention number from a wrongly composed head is a comparison against a model
that was never benchmarked, and there is no way to tell that from the curve.

`eval.detectors.dinov3.DINOv3MLPDetector` is built with the seam already in
it, so this class delegates rather than reconstructs:

    trunk -> detector.trunk    frozen DINOv3 ViT-S/16, pooled -> (B, D)
    head  -> detector.head     LayerNorm -> MLP -> one logit

`head(trunk(x)) == detector(x)` is then true by construction -- `forward` on the
detector is literally `self.head(self.trunk(x))`. `verify_split` still runs,
because "true by construction" is a claim about code that someone will edit.

Layout is `vector`, deliberately
--------------------------------
The pooled descriptor is one embedding per image, so the adapter's gate is a
single `(D,)` vector and there is no per-block damage profile to plot. That
forfeits the per-layer interpretability figure a `layers` seam would give, and
it is the right trade for a PoC: it makes the cache 768 bytes per image per view instead of 48 KB (a factor
of 64), which is what lets the whole pipeline -- render, stage 1, stage 2, eval --
run end to end in minutes on a laptop with no GPU.

The `layers` variant is a small change when it is wanted: emit the per-block CLS
tokens from `output_hidden_states=True` as `(B, 12, 384)` and give the head a
RINE-style importance weighting over them. Everything downstream of the split
already handles that layout -- `grace_adapter.models.factory` picks the `(L, D)` gate off
`FeatureSpec.layout` alone.
"""

import torch

from eval.splits.base import FeatureSpec, SplitDetector
from eval.splits.verify import verify_split


class DINOv3Split(SplitDetector):
    """Cut the DINOv3 probe detector at its pooled feature vector."""

    def __init__(self, detector, verify: bool = True):
        super().__init__(detector)
        for attr in ("trunk", "head", "feature_dim"):
            if not hasattr(detector, attr):
                raise TypeError(
                    f"DINOv3Split expects a DINOv3MLPDetector (it needs .{attr}); "
                    f"got {type(detector).__name__}. This split delegates to a "
                    f"detector that already exposes its seam -- it does not "
                    f"reconstruct one."
                )
        # Not an error: rendering a cache needs only the trunk, and the trunk is
        # frozen pretrained weights whatever the head holds. Scoring with an
        # untrained head, on the other hand, produces a ~0.5 AUC that looks like
        # a failed adapter rather than a missing stage 0.
        if getattr(detector, "head_untrained", False):
            import warnings

            warnings.warn(
                f"{detector.name} has a randomly initialized head "
                "(head_checkpoint: null). Trunk features are still valid, so a "
                "cache rendered from this is fine -- but any logit, AUC or "
                "head-Jacobian computed from it is noise. Run "
                "scripts/train_probe.py first.",
                stacklevel=2,
            )

        if verify:
            verify_split(self)

    @property
    def feature_spec(self) -> FeatureSpec:
        # Read off the detector, never hardcoded: `pool` changes the width by 2x
        # and a ViT-B mirror changes it again.
        return FeatureSpec(layout="vector", shape=(self.detector.feature_dim,))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.detector.trunk(x)

    def head(self, f: torch.Tensor) -> torch.Tensor:
        return self.detector.head(f)

    def head_modules(self) -> dict:
        """For `verify_split`'s failure message. One module, by construction."""
        return {"head_module": self.detector.head_module}
