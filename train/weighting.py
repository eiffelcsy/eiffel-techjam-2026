"""Spend adapter capacity where it changes the decision.

Plain MSE treats every feature direction as equally worth fixing. The head does
not: it is a linear map or a shallow MLP into one scalar, so only the component
of the error inside its sensitive subspace can move the logit, and everything
orthogonal is adapter capacity spent on nothing an AUC can see.

The head maps features to one scalar, so its Jacobian is a gradient vector with
the same shape as the feature:

    j_i = ∇_f h(f) |_{f = f_clean_i}

**For a linear head this is exactly the constant `w`**, so one implementation
covers the linear-head and MLP-head cases with no branch -- which is the whole
reason to express the weighting this way rather than as "project onto w".

    e     = f_adapted − f_clean
    L_err = (1−ε)·mean_B[(ĵ·e)²]  +  ε·mean[e²]          ĵ = j/‖j‖

The first term is, to first order, the squared error *in the logit*. The second
is an isotropic floor that keeps the objective well-posed and preserves the
magnitude information a nonlinear head may use downstream. Written as a blend so
that `ε = 1` is exactly `F.mse_loss` -- the plain-MSE ablation is then one config
key and provably the same objective GRACE v1 had.

Relation to `head_kl`: that term is the *exact* version of the first term
(finite difference through the real head rather than a first-order expansion),
but it only ever observes the scalar and so says nothing about which feature
directions to fix. The weighting shapes the whole residual with the same
geometry. Both are kept; `lam_kl` is demoted to 0.1 by default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def head_gradient(head: nn.Module, f: torch.Tensor) -> torch.Tensor:
    """∇_f head(f), detached, same shape as `f`.

    `head(f).sum()` is safe: sample i's logit depends only on sample i's
    features, so the summed backward yields per-sample gradients rather than
    mixing them. That holds for any batch-independent head -- true here because
    detectors are frozen and in eval mode, so BatchNorm uses running statistics.

    Runs on its own graph. The gradient is a property of the clean features and
    the frozen head, and must not leak into the adapter's graph.
    """
    with torch.enable_grad():
        x = f.detach().requires_grad_(True)
        out = head(x)
        (grad,) = torch.autograd.grad(out.sum(), x)
    return grad.detach()


def decision_weighted_error(
    e: torch.Tensor, j: torch.Tensor, eps_iso: float = 0.05
) -> torch.Tensor:
    """Squared error, weighted toward the head's sensitive direction.

    `eps_iso = 1.0` reduces exactly to `F.mse_loss(f_adapted, f_clean)`.
    """
    if not 0.0 <= eps_iso <= 1.0:
        raise ValueError(f"eps_iso must be in [0, 1], got {eps_iso}")
    iso = e.pow(2).mean()
    if eps_iso == 1.0:
        return iso
    feat_dims = tuple(range(1, e.ndim))
    j_hat = j / (j.flatten(1).norm(dim=1).clamp_min(EPS).reshape(-1, *(1,) * (e.ndim - 1)))
    parallel = (j_hat * e).sum(dim=feat_dims).pow(2).mean()
    return (1.0 - eps_iso) * parallel + eps_iso * iso


def logit_error(head: nn.Module, f_adapted: torch.Tensor, f_clean: torch.Tensor) -> torch.Tensor:
    """The quantity `decision_weighted_error` approximates, computed exactly.

    Not used as a loss -- `head_kl` fills that role -- but reported at validation
    so the first-order approximation can be checked against the truth it stands
    in for. If they diverge, `eps_iso` is doing more work than intended.
    """
    with torch.no_grad():
        return F.mse_loss(head(f_adapted), head(f_clean))
