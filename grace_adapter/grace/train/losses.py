"""What "recovered" means.

    L = L_align + λ_kl·L_head_KL + λ_sev·L_severity

Every term above is **label-free**: the targets are the detector's own clean-view
features and the degradation sampler's own metadata. `supervised_bce` is the one
exception and belongs to stage 2 (`grace.models.discrepancy`), not here.

Every term is also logged separately. A retention gain from the head-KL term is a
different result than a gain from the alignment term, and the aggregate number
cannot tell you which happened.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from grace.train.weighting import decision_weighted_error


def _scalar(t: torch.Tensor) -> float:
    """Detach before float(): logging must never hold a graph alive."""
    return float(t.detach())


def alignment_loss(
    f_adapted: torch.Tensor,
    f_clean: torch.Tensor,
    j: torch.Tensor | None = None,
    w_cos: float = 1.0,
    w_err: float = 1.0,
    eps_iso: float = 0.05,
    weighting: str = "jacobian",
) -> torch.Tensor:
    """The primary objective: pull corrected features onto the clean-view target.

    Cosine on L2-normalized features, plus a squared-error term on the raw ones.
    Both, deliberately: the head is sensitive to feature *magnitude*, so a
    pure-cosine objective is free to rescale everything and quietly break the
    frozen head, while a pure squared-error objective is dominated by whichever
    channels happen to have the largest variance.

    `weighting="jacobian"` sends the error term through
    `decision_weighted_error`; `"none"` is plain MSE, i.e. exactly the GRACE v1
    objective, kept as the ablation.
    """
    a = F.normalize(f_adapted, dim=-1)
    c = F.normalize(f_clean, dim=-1)
    cos = (1 - (a * c).sum(-1)).mean()

    e = f_adapted - f_clean
    if weighting == "none" or j is None:
        err = e.pow(2).mean()
    elif weighting == "jacobian":
        err = decision_weighted_error(e, j, eps_iso)
    else:
        raise ValueError(f"weighting must be 'none' or 'jacobian', got {weighting!r}")
    return w_cos * cos + w_err * err


def head_kl(
    head: nn.Module, f_adapted: torch.Tensor, f_clean: torch.Tensor, T: float = 2.0
) -> torch.Tensor:
    """Align through the frozen head, not only in feature space.

    The exact counterpart to the Jacobian-weighted error term: a finite
    difference through the real head rather than a first-order expansion of it.
    Costs one linear layer. The clean logits are detached -- the teacher is a
    constant, never a gradient path.
    """
    p = torch.sigmoid(head(f_clean).detach() / T)
    q = torch.sigmoid(head(f_adapted) / T)
    return F.binary_cross_entropy(q, p) * T * T


def severity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Auxiliary regression onto the sampler's own severity. Label-free."""
    return F.mse_loss(pred, target)


def supervised_bce(logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Stage 2 only. The one place GRACE uses image labels."""
    return F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels.float())


def total_loss(
    *,
    head: nn.Module,
    f_adapted: torch.Tensor,
    f_clean: torch.Tensor,
    j: torch.Tensor | None,
    severity_pred: torch.Tensor | None,
    severity_target: torch.Tensor | None,
    cfg,
    diagnostics: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Sum the terms and return (loss, per-term scalars for the log).

    `diagnostics` reports every term regardless of its weight, so a `lam_X: 0`
    ablation can still be read on the term it ablated. It never changes `loss`.
    """
    align = alignment_loss(
        f_adapted, f_clean, j,
        w_cos=cfg.w_cos, w_err=cfg.w_err,
        eps_iso=cfg.eps_iso, weighting=cfg.weighting,
    )

    terms = {"align": _scalar(align)}
    loss = align

    # Each term is gated on its own `lam > 0`, so an ablation that zeroes a
    # weight also stops logging the term it ablates -- exactly the quantity the
    # ablation is about. Under `diagnostics` the term is still computed and
    # reported; only its contribution to `loss` stays off. The caller sets this
    # on logging steps alone, so a disabled term costs nothing on the rest.
    if cfg.lam_kl > 0 or diagnostics:
        kl = head_kl(head, f_adapted, f_clean, cfg.kl_temperature)
        if cfg.lam_kl > 0:
            loss = loss + cfg.lam_kl * kl
        terms["head_kl"] = _scalar(kl)

    if (cfg.lam_sev > 0 or diagnostics) and severity_pred is not None and severity_target is not None:
        sev = severity_loss(severity_pred, severity_target)
        if cfg.lam_sev > 0:
            loss = loss + cfg.lam_sev * sev
        terms["severity"] = _scalar(sev)

    terms["total"] = _scalar(loss)
    return loss, terms
