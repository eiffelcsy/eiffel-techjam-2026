"""Metrics.

Four questions, in order of importance:

  1. Does the detector separate real from generated?      -> AUC
  2. How much survives degradation, level by level?       -> retention
  3. Which way does it fail when it fails?                -> FP / FN split
  4. Is a composition worse than its parts predict?       -> interaction gap

The threshold is always picked on clean data and then applied unchanged to
every degraded condition. That is the deployment reality, and it exposes
calibration drift that AUC hides -- a detector can keep its ranking (AUC holds)
while every degraded image slides to one side of a fixed threshold.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

RETENTION_FLOOR = 0.5
"""Retention below which a level counts as failed.

The `operating_envelope` in the results schema is the deepest composition level
still at or above this. 0.5 means "at least half the chance-corrected clean
skill survives"; it is a reporting convention, not a derived quantity, and
the run config's `retention_floor` overrides it.
"""


@dataclass
class ErrorBreakdown:
    """Counts and rates at a fixed threshold. `fp` = a real image called AI."""

    tp: int
    fp: int
    tn: int
    fn: int
    fpr: float          # reals called generated
    fnr: float          # generated called real
    precision: float
    recall: float
    f1: float
    accuracy: float

    def as_dict(self) -> dict:
        return asdict(self)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores)))


def retention(auc_degraded: float, auc_clean: float) -> float:
    """robust AUC / clean AUC, on the chance-corrected scale:

        (auc_deg - 0.5) / (auc_clean - 0.5)

    so that a detector dropping to chance scores 0, not 0.5.
    """
    denom = auc_clean - 0.5
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float((auc_degraded - 0.5) / denom)


def threshold_from_clean(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pick the operating point on clean data (max F1); apply it everywhere."""
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(labels), np.asarray(scores)
    )
    # precision_recall_curve returns one more point than thresholds
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def error_breakdown(scores: np.ndarray, labels: np.ndarray, thr: float) -> ErrorBreakdown:
    labels = np.asarray(labels).astype(int)
    pred = (np.asarray(scores) >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    def _div(a, b):
        return float(a / b) if b else 0.0

    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    return ErrorBreakdown(
        tp=tp, fp=fp, tn=tn, fn=fn,
        fpr=_div(fp, fp + tn),
        fnr=_div(fn, fn + tp),
        precision=precision,
        recall=recall,
        f1=_div(2 * precision * recall, precision + recall),
        accuracy=_div(tp + tn, tp + fp + tn + fn),
    )


def score_shift(scores_clean: np.ndarray, scores_degraded: np.ndarray) -> float:
    """Mean signed change in score, paired by image index.

    Tells you the *direction* a transform pushes a detector: blur that makes
    everything look generated is a different failure from blur that erases the
    generation traces.
    """
    return float(np.mean(np.asarray(scores_degraded) - np.asarray(scores_clean)))


def bootstrap_ci(
    fn, *arrays, groups=None, n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI, resampling images (not rows) so replicates of the same
    image stay together. The composed levels are Monte-Carlo estimates; without
    an interval you cannot tell a real gap between detectors from the draw.

    `groups` is the per-row image index. Omitting it resamples rows, which is
    only correct when each image contributes exactly one row.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n_rows = len(arrays[0])

    if groups is None:
        rows_of = np.arange(n_rows).reshape(-1, 1)
        n_units = n_rows
    else:
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        rows_of = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
        n_units = len(rows_of)

    stats = []
    for _ in range(n):
        pick = rng.integers(0, n_units, n_units)
        rows = np.concatenate([np.atleast_1d(rows_of[p]) for p in pick])
        value = fn(*(a[rows] for a in arrays))
        if np.isfinite(value):
            stats.append(value)

    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# --- what the level structure buys you ---------------------------------------

def predicted_composed_retention(
    single_retentions: dict[str, float],
    recipes: list[list[str]],
) -> float:
    """Retention a composed level would show if transforms did not interact.

    The independence baseline: each recipe's retention is the product of its
    steps' L1 retentions, averaged over the sampled recipes. Deliberately
    naive -- it exists only to be compared against the measured value.

    `single_retentions` is keyed by transform name (the L1 marginal, pooled
    over that transform's parameter grid).
    """
    if not recipes:
        return float("nan")
    products = [
        float(np.prod([single_retentions.get(t, 1.0) for t in recipe]))
        for recipe in recipes
    ]
    return float(np.mean(products))


def interaction_gap(measured: float, predicted: float) -> float:
    """measured - predicted composed retention.

    Negative means the composition hurts more than its parts do separately,
    which is the case worth reporting: it says single-transform benchmarks
    overstate this detector's robustness, and by how much.
    """
    return float(measured - predicted)


def operating_envelope(retention_by_level: dict[str, float], floor: float = RETENTION_FLOOR) -> int:
    """Deepest composition level still at or above the retention floor.

    0 means even single transforms push the detector under the floor.
    """
    deepest = 0
    for level in (1, 2, 3):
        value = retention_by_level.get(f"L{level}")
        if value is None or not np.isfinite(value) or value < floor:
            break
        deepest = level
    return deepest


def per_recipe_breakdown(scores, labels, recipes, thr) -> pd.DataFrame:
    """AUC and error rates grouped by the transform *set* in each recipe.

    Only meaningful at L2/L3, where each image has its own recipe. Surfaces
    which pairs are lethal (typically a low-pass step plus compression) and
    whether order mattered for a given pair.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    keys = [tuple(sorted(set(r))) for r in recipes]

    rows = []
    for key in sorted(set(keys)):
        mask = np.array([k == key for k in keys])
        errors = error_breakdown(scores[mask], labels[mask], thr)
        rows.append({
            "transforms": list(key),
            "n": int(mask.sum()),
            "auc": roc_auc(scores[mask], labels[mask]),
            "fpr": errors.fpr,
            "fnr": errors.fnr,
        })
    return pd.DataFrame(rows)
