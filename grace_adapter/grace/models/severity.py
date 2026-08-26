"""Severity prediction -- an auxiliary task whose labels are free and exact.

The target does not come from annotation, and it does not come from the day-3
approximation either. The transform grids in
`eval_pipeline/pipeline/degrade/ops.py` are **ordered mild -> severe**, so a
step's severity is the normalized rank of its parameter within its own grid, and
a recipe's severity combines that with how many steps were composed. The
degradation sampler already knows both, so the target is written into the cache
alongside the features at render time. See
`grace.cache.schedule.EpochSchedule.severity_for`.

Consequence worth being explicit about: **severity conditioning does not cost the
label-free claim.** The labels are the sampler's own metadata, not image labels.
Only the stage-2 discrepancy head uses real labels.

At inference the severity is *predicted*, never given, so training must not
always condition on the ground truth or the adapter learns to trust an input it
will not have. `grace.train.loop` feeds the prediction on half the steps.
"""

import torch
import torch.nn as nn


class SeverityHead(nn.Module):
    """Degraded features -> scalar corruption severity in [0, 1].

    Pools the group axis before the MLP: severity is a property of the image, so
    a per-layer or per-token estimate would be predicting the same number many
    times.
    """

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if f.ndim > 2:
            f = f.flatten(1, -2).mean(dim=1)
        return torch.sigmoid(self.net(f)).squeeze(-1)
