"""GAPL split -- vector layout, cheapest cache, prototype in the head.

A LoRA-wrapped CLIP backbone into a Linear(128 -> 1). The seam is the pooled
embedding before that linear.

Two wrinkles:

  * `load_prototype`: GAPL's decision is partly a distance to stored class
    prototypes, so the "head" here is prototype comparison *plus* linear, and the
    prototype tensor is frozen state, not a buffer to re-estimate. Nothing in
    GRACE updates it.
  * The prototype makes the head's input-Jacobian genuinely non-constant, unlike
    a bare linear head. That is a feature for `grace.train.weighting` -- it is the
    case that justifies computing the gradient rather than reading off `w` -- and
    a reason to include GAPL in the weighting ablation.

Vector layout at 2 KB per image per view, so this is the cheapest detector to
cache and the fastest way to get the whole cache/train/eval loop running
end-to-end once its clone is present.

VERIFY AGAINST THE CLONE -- see `grace.splits.verify`.
"""

import torch

from grace.splits.base import FeatureSpec, SplitDetector
from grace.splits.verify import verify_split


class GAPLSplit(SplitDetector):
    """Cut GAPL at the pooled backbone embedding."""

    def __init__(self, detector, verify: bool = True):
        super().__init__(detector)
        if verify:
            verify_split(self)

    @property
    def feature_spec(self) -> FeatureSpec:
        raise NotImplementedError("needs third_party/GAPL -- see the module docstring")

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "GAPLSplit.trunk needs third_party/GAPL: cut GAPLModel between the "
            "LoRA-wrapped backbone's pooled output and the prototype/linear head."
        )

    def head(self, f: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("see GAPLSplit.trunk")
