"""DINOv3 split

`eval.detectors.dinov3.DINOv3MLPDetector` is built with the seam already in
it, so this class delegates rather than reconstructs:

    trunk -> detector.trunk    frozen DINOv3 ViT-S/16, pooled -> (B, D)
    head  -> detector.head     LayerNorm -> MLP -> one logit

`head(trunk(x)) == detector(x)` is then true by construction -- `forward` on the
detector is literally `self.head(self.trunk(x))`. `verify_split` still runs,
because "true by construction" is a claim about code that someone will edit.
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
