"""What "recovered" means.

    L = L_align + λ_sw·L_sw + λ_id·L_identity + λ_kl·L_head_KL + λ_sev·L_severity

Every term above is **label-free**: the targets are the detector's own clean-view
features and the degradation sampler's own metadata. `supervised_bce` is the one
exception and belongs to stage 2 (`grace.models.discrepancy`), not here.

Every term is also logged separately. A retention gain from the distribution term
is a different result than a gain from the alignment term, and the aggregate
number cannot tell you which happened.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from grace.train.weighting import decision_weighted_error

EPS = 1e-8


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


def sliced_wasserstein(
    a: torch.Tensor, b: torch.Tensor, n_proj: int = 64, generator=None
) -> torch.Tensor:
    """Match the *distribution* of adapted features to that of clean features.

    Point-wise alignment asks each corrected feature to sit on its own target and
    is satisfied by a conditional mean, which is systematically under-dispersed:
    the batch ends up in a tighter cloud than real clean features form, and the
    frozen head's operating point was calibrated on the wider one. This is the
    feature-space analogue of the perceptual/adversarial term in
    super-resolution, and it is the term that makes the adapter's noise input
    worth having -- without it, `z` is ignored and posterior sampling collapses.

    Sliced Wasserstein rather than MMD or a discriminator: random projections,
    sort, L2. Six lines, one hyperparameter, and no adversarial stability risk.

    Batch-level statistic, so it needs a real batch (256+ is the default; these
    are features, not images). Because `a` and `b` hold the *same images*, this
    is a matched comparison and strictly stronger than the usual unpaired form.

    For grouped layouts the projections are shared but the sort is per group:
    flattening would blend per-layer statistics that the per-layer gate exists to
    keep apart.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    d = a.shape[-1]
    p = torch.randn(d, n_proj, device=a.device, dtype=a.dtype, generator=generator)
    p = p / p.norm(dim=0, keepdim=True).clamp_min(EPS)
    pa = (a @ p).sort(dim=0).values
    pb = (b @ p).sort(dim=0).values
    return (pa - pb).pow(2).mean()


def identity_loss(
    adapter: nn.Module, f_clean: torch.Tensor, severity: torch.Tensor | None = None
) -> torch.Tensor:
    """On genuinely clean features, the adapter must do nothing.

    Costs no extra trunk compute -- it is `adapter(f_clean)` against `f_clean`,
    and `f_clean` is already in hand from the cache. Deterministic pass on
    purpose: identity should hold for the mean correction, and drawing noise here
    would make the term needlessly high-variance.

    Note this is the *explicit* constraint. The implicit one does more work: ~15%
    of training samples are level-0, where the target simply equals the input.
    See `grace.cache.schedule.DEFAULT_LEVEL_WEIGHTS`.
    """
    return F.mse_loss(adapter(f_clean, severity=severity), f_clean)


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
    adapter: nn.Module,
    head: nn.Module,
    f_adapted: torch.Tensor,
    f_clean: torch.Tensor,
    j: torch.Tensor | None,
    severity_pred: torch.Tensor | None,
    severity_target: torch.Tensor | None,
    cfg,
) -> tuple[torch.Tensor, dict]:
    """Sum the terms and return (loss, per-term scalars for the log).

    `f_adapted` may carry a leading sample axis `(k, B, ...)` from posterior
    sampling; the point-wise terms are averaged over it and the distributional
    term is computed on the pooled draws, which is what gives the noise
    something to do.
    """
    sampled = f_adapted.ndim == f_clean.ndim + 1
    draws = f_adapted if sampled else f_adapted.unsqueeze(0)
    k = draws.shape[0]

    align = sum(
        alignment_loss(
            draws[i], f_clean, j,
            w_cos=cfg.w_cos, w_err=cfg.w_err,
            eps_iso=cfg.eps_iso, weighting=cfg.weighting,
        )
        for i in range(k)
    ) / k

    terms = {"align": _scalar(align)}
    loss = align

    if cfg.lam_sw > 0:
        # Pooled over draws against the clean batch tiled to match, so a
        # collapsed posterior is penalised for under-dispersion.
        sw = sliced_wasserstein(
            draws.flatten(0, 1), f_clean.repeat(k, *(1,) * (f_clean.ndim - 1)), cfg.n_proj
        )
        loss = loss + cfg.lam_sw * sw
        terms["sw"] = _scalar(sw)

    if cfg.lam_id > 0:
        ident = identity_loss(adapter, f_clean, severity_target)
        loss = loss + cfg.lam_id * ident
        terms["identity"] = _scalar(ident)

    if cfg.lam_kl > 0:
        kl = sum(head_kl(head, draws[i], f_clean, cfg.kl_temperature) for i in range(k)) / k
        loss = loss + cfg.lam_kl * kl
        terms["head_kl"] = _scalar(kl)

    if cfg.lam_sev > 0 and severity_pred is not None and severity_target is not None:
        sev = severity_loss(severity_pred, severity_target)
        loss = loss + cfg.lam_sev * sev
        terms["severity"] = _scalar(sev)

    terms["total"] = _scalar(loss)
    return loss, terms
