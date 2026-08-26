"""Measurements that decide whether the objective is doing what it claims.

Three questions, each cheap enough to log continuously:

  1. Is the correction pointed at the decision, or wasted?   -> decision_alignment
  2. Does drift carry forensic signal we are erasing?        -> drift_asymmetry
  3. Is the posterior actually stochastic?                    -> posterior_spread

None of these is a loss. They exist so that a result can be explained rather than
just reported, and (2) in particular is run *before* any training, on the cache
alone, by `scripts/analyze_drift.py`.
"""

import numpy as np
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


def drift(f_deg: torch.Tensor, f_clean: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample drift magnitude and direction, relative to feature scale.

    Relative, because raw ‖·‖ is not comparable across detectors or across the
    layers of one detector.
    """
    d = _flat(f_deg - f_clean)
    scale = _flat(f_clean).norm(dim=1).clamp_min(EPS)
    return {
        "relative": d.norm(dim=1) / scale,
        "cosine": F.cosine_similarity(_flat(f_deg), _flat(f_clean), dim=1),
    }


def drift_asymmetry(
    f_deg: torch.Tensor,
    f_clean: torch.Tensor,
    labels: torch.Tensor,
    j: torch.Tensor | None = None,
) -> dict[str, float]:
    """RA-Det's claim, tested on this data: do generated images drift further?

    Returns the fake-minus-real gap in relative drift, and -- when the head
    gradient is supplied -- the same gap decomposed into the component parallel
    to the decision direction and the component orthogonal to it.

    That decomposition is what determines whether the discrepancy branch can
    work. Drift that is large but entirely orthogonal to the head's sensitive
    subspace is invisible to the frozen head, which is precisely why an
    *auxiliary* head reading Δ can recover signal the main head cannot -- and it
    is also why the label-free objective can erase that signal without its own
    loss noticing.
    """
    lab = labels.reshape(-1).bool()
    if lab.all() or (~lab).all():
        return {"n_real": int((~lab).sum()), "n_fake": int(lab.sum())}

    stats = drift(f_deg, f_clean)
    out = {
        "drift_real": float(stats["relative"][~lab].mean()),
        "drift_fake": float(stats["relative"][lab].mean()),
        "n_real": int((~lab).sum()),
        "n_fake": int(lab.sum()),
    }
    out["asymmetry"] = out["drift_fake"] - out["drift_real"]

    if j is not None:
        d = _flat(f_deg - f_clean)
        j_hat = F.normalize(_flat(j), dim=1)
        para = (d * j_hat).sum(dim=1).abs()
        orth = (d - para.unsqueeze(1) * j_hat).norm(dim=1)
        out["parallel_asymmetry"] = float(para[lab].mean() - para[~lab].mean())
        out["orthogonal_asymmetry"] = float(orth[lab].mean() - orth[~lab].mean())
        out["parallel_fraction"] = float((para / d.norm(dim=1).clamp_min(EPS)).mean())
    return out


def posterior_spread(logits: torch.Tensor) -> float:
    """Std of the logit across posterior draws, averaged over the batch.

    The posterior-collapse tripwire. Under point-wise reconstruction losses alone
    the optimal stochastic policy is to ignore `z`, and this reads ~0. That is a
    reportable negative result about the objective, not a bug to paper over --
    see `grace.train.losses.sliced_wasserstein`.
    """
    if logits.ndim < 2 or logits.shape[0] < 2:
        return 0.0
    return float(logits.std(dim=0).mean())


def bootstrap_gap(values: np.ndarray, labels: np.ndarray, n: int = 1000, seed: int = 0):
    """Percentile CI on the fake-minus-real gap in `values`.

    Resamples images, matching `pipeline.eval.metrics.bootstrap_ci`'s convention
    so the intervals here are comparable with the harness's.
    """
    rng = np.random.default_rng(seed)
    values, labels = np.asarray(values), np.asarray(labels).astype(bool)
    gaps = np.empty(n)
    idx = np.arange(len(values))
    for i in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        v, lb = values[pick], labels[pick]
        gaps[i] = (
            v[lb].mean() - v[~lb].mean() if lb.any() and (~lb).any() else np.nan
        )
    return float(np.nanpercentile(gaps, 2.5)), float(np.nanpercentile(gaps, 97.5))
