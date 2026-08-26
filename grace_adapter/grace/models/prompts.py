"""FUTURE -- degradation prompts. BLUEPRINT ONLY, nothing implemented.

Replaces the scalar-severity FiLM in `grace.models.severity`, which is thin: one
number cannot distinguish "JPEG at quality 30" from "blur at sigma 2.0", yet
those need different corrections, not merely different magnitudes of the same
correction.

The all-in-one restoration literature solved this. PromptIR carries a bank of
learnable degradation prompts and selects over them by soft attention on the
input features; AirNet obtains the degradation embedding **contrastively and
without labels**, by pulling together two crops of the same degraded image and
pushing apart different degradations. The second point matters here more than
the first: a contrastive degradation embedding fits GRACE's label-free framing
better than the supervised severity target does, and would let the conditioning
drop the one place the primary objective currently leans on sampler metadata.

It is also a cleaner novelty statement. "Degradation prompts for feature-space
restoration" says what is new; "FiLM on a gate" says what is standard.

Intended shape:

    e     = encoder(f_deg)                  # degradation embedding, contrastive
    w     = softmax(e @ P.T / tau)          # soft attention over the bank
    p     = w @ P                           # the mixture prompt
    corr  = adapter_block(f_deg, prompt=p)  # prompt conditions the bottleneck

Same parameter budget as the FiLM path at a bank of ~8-16 prompts. Drops in at
one call site: `GatedResidualAdapter.gate(severity=...)` becomes
`gate(prompt=...)`, which is why the severity scalar was kept behind that
interface rather than threaded through the loop.

Evaluation hook that makes this worth doing: the attention weights `w` are a
soft *classification of the degradation*, obtained without degradation labels.
Compare them against the known recipe from `recipes.parquet` and the confusion
matrix is a free figure -- and a check that the prompts learned the corruption
family rather than partitioning on image content.
"""

import torch
import torch.nn as nn


class PromptBank(nn.Module):
    """FUTURE. A bank of learnable degradation prompts, selected by soft attention."""

    def __init__(self, dim: int, n_prompts: int = 8, prompt_dim: int = 64, tau: float = 1.0):
        raise NotImplementedError("FUTURE -- see module docstring")

    def forward(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (mixture prompt, attention weights). The weights are the figure."""
        raise NotImplementedError("FUTURE -- see module docstring")


class DegradationEncoder(nn.Module):
    """FUTURE. AirNet-style contrastive embedding of *how* an image was degraded.

    Trained with InfoNCE over pairs sharing a recipe. Label-free: the positives
    come from the degradation sampler, never from image labels.
    """

    def __init__(self, dim: int, out_dim: int = 64):
        raise NotImplementedError("FUTURE -- see module docstring")

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FUTURE -- see module docstring")


def contrastive_degradation_loss(e_a: torch.Tensor, e_b: torch.Tensor, tau: float = 0.07):
    """FUTURE. InfoNCE over two views sharing a degradation recipe."""
    raise NotImplementedError("FUTURE -- see module docstring")
