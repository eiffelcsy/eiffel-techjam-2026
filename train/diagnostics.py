"""Measurements that decide whether the objective is doing what it claims.

One question, cheap enough to log continuously:

  Is the correction pointed at the decision, or wasted?   -> decision_alignment

It is not a loss. It exists so that a result can be explained rather than just
reported.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(1)


def decision_alignment(
    f_adapted: torch.Tensor, f_deg: torch.Tensor, j: torch.Tensor
) -> torch.Tensor:
    """cos(Δ, j) per sample, where Δ = f_adapted − f_deg is the correction.

    **The figure.** Only the component of the correction inside the head's
    sensitive subspace can move AUC. If a plain-MSE run sits near cos ≈ 0, the
    adapter is spending nearly all of its capacity on directions the head cannot
    see -- direct evidence that the objective misallocates capacity, and the
    empirical motivation for `weighting: jacobian` rather than an asserted one.

    Report `|cos|`, not `cos`: a correction that moves *against* the decision
    direction is still inside the sensitive subspace. Whether it moves the right
    way is what AUC is for.
    """
    return F.cosine_similarity(_flat(f_adapted - f_deg), _flat(j), dim=1)


def energy_fraction(f_adapted: torch.Tensor, f_deg: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Fraction of squared correction energy lying in the decision direction.

    `cos²`, stated as the quantity a reader actually wants: "3% of what the
    adapter did could possibly have changed the answer".
    """
    return decision_alignment(f_adapted, f_deg, j).pow(2)
