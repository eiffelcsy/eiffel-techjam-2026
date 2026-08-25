"""The evaluation loop.

For each detector: score the fixed eval subset once per condition -- clean
first, then the 14 L1 grid conditions, then the L2 and L3 composed conditions
(n_replicates draws each) -- and compute AUC, retention, and the error
breakdown against the clean threshold.

Every condition scores the same images in the same order, so clean and
degraded scores are paired per image. Per-image recipes are logged at L2/L3;
without them the composed levels are just a number, with them they are an
analysable sample of the composition space.

A run is one detector against a set of datasets. The detector is loaded once
and reused, and each dataset is scored independently -- its own clean
threshold, its own retention denominator -- into
results/{run_id}__{detector}__{dataset}.json.

Result schema -- fix it now, the report reads this and nothing else:

{
  "run_id": str,
  "detector": str,
  "dataset": str,
  "n_images": int, "n_real": int, "n_fake": int,
  "clean_threshold": float,

  "levels": {                              # the headline structure
    "L0_clean":  {"auc": float, "retention": 1.0, "errors": {...}},
    "L1_single": {"auc": float, "retention": float, "errors": {...},
                  "auc_ci": [lo, hi]},     # pooled over the 14 grid conditions
    "L2_pair":   {"auc": float, "retention": float, "errors": {...},
                  "auc_ci": [lo, hi],
                  "predicted_retention": float,   # from L1 marginals
                  "interaction_gap": float},
    "L3_multi":  {... same as L2 ...}
  },

  "conditions": {                          # L1 only: one entry per grid point
    "jpeg/quality=30": {
      "transform": "jpeg", "group": "compression",
      "param_name": "quality", "param": 30,
      "auc": float, "retention": float, "score_shift": float,
      "errors": {...ErrorBreakdown...}
    },
    ...
  },

  "by_transform": {                        # L1 collapsed over each grid
    "jpeg": {"mean_auc": float, "worst_auc": float,
             "mean_retention": float, "worst_retention": float,
             "worst_param": 30},
    ...
  },

  "recipes": {                             # L2/L3, grouped by transform set
    "L2": [{"transforms": ["gaussian_blur", "jpeg"], "n": int,
            "auc": float, "retention": float, "fpr": float, "fnr": float}, ...],
    "L3": [...]
  },

  "summary": {
    "clean_auc": float,
    "retention_by_level": {"L1": float, "L2": float, "L3": float},
    "worst_condition": "jpeg/quality=30",
    "worst_recipe": ["gaussian_blur", "jpeg", "resize"],
    "operating_envelope": int              # deepest level still above the
                                           # retention floor (see report)
  }
}
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.data.dataset import AIGCDataset, collate
from pipeline.data.manifest import load_manifest, sample_eval_subset
from pipeline.degrade.conditions import build_conditions, load_grid
from pipeline.degrade.ops import TRANSFORMS
from pipeline.detectors import build_detector, resolve_device
from pipeline.eval import metrics as M
from pipeline.utils.io import write_json

LEVEL_KEYS = {0: "L0_clean", 1: "L1_single", 2: "L2_pair", 3: "L3_multi"}
N_BOOTSTRAP = 200
"""Bootstrap resamples for the pooled level CIs.

200 is enough for a 95% percentile interval at the precision reported here, and
keeps the aggregation cost negligible next to the forward passes.
"""


def score_dataset(detector, dataset, batch_size: int, num_workers: int, device) -> tuple:
    """Run the detector over one condition.

    Returns (scores, labels, records) in manifest order. `records` is the
    per-image meta -- index, label, recipe, transforms -- constant at L0/L1,
    one draw per image at L2/L3.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,          # manifest order is the pairing across conditions
        collate_fn=collate,
    )
    scores, records = [], []
    for batch, metas in tqdm(loader, desc=dataset.condition.id, leave=False):
        scores.append(detector.score(batch.to(device)).float().cpu().numpy())
        records.extend(metas)
    scores = np.concatenate(scores) if scores else np.array([])
    labels = np.array([r["label"] for r in records], dtype=int)
    return scores, labels, records


def aggregate_by_transform(conditions: dict) -> dict:
    """Collapse each L1 transform grid into mean / worst rows."""
    by: dict[str, dict] = {}
    for entry in conditions.values():
        by.setdefault(entry["transform"], []).append(entry)

    out = {}
    for name, entries in by.items():
        aucs = [e["auc"] for e in entries]
        rets = [e["retention"] for e in entries]
        # Entries arrive in grid order, which is mild -> severe. Ties break
        # toward the later (harsher) parameter: when a detector is flat across
        # a grid, the honest "worst case" is the strongest setting it survived,
        # not the weakest one that happened to be scored first.
        worst = entries[min(range(len(entries)), key=lambda i: (entries[i]["auc"], -i))]
        out[name] = {
            "mean_auc": float(np.mean(aucs)),
            "worst_auc": float(np.min(aucs)),
            "mean_retention": float(np.mean(rets)),
            "worst_retention": float(np.min(rets)),
            "worst_param": worst["param"],
        }
    return out


def aggregate_by_level(conditions: dict, level_results: dict) -> dict:
    """Pool each level into one row, and for L2/L3 attach the independence
    prediction from the L1 marginals plus the resulting interaction gap."""
    marginals = {
        name: row["mean_retention"]
        for name, row in aggregate_by_transform(conditions).items()
    } if conditions else {}

    levels = {}
    for level, result in sorted(level_results.items()):
        recipes = result.pop("_recipes", [])
        entry = {k: v for k, v in result.items() if not k.startswith("_")}
        if level >= 2:
            predicted = M.predicted_composed_retention(marginals, recipes)
            entry["predicted_retention"] = predicted
            entry["interaction_gap"] = M.interaction_gap(entry["retention"], predicted)
        levels[LEVEL_KEYS[level]] = entry
    return levels


def evaluate_detector(detector, manifest, conditions, device="auto",
                      batch_size: int = 32, num_workers: int = 4,
                      retention_floor: float = M.RETENTION_FLOOR) -> dict:
    """Score every condition and assemble the dict documented above.

    Clean is scored first: its scores set the threshold, the retention
    denominator, and the pairing for score_shift.
    """
    device = resolve_device(device) if isinstance(device, str) else device
    ordered = sorted(conditions, key=lambda c: (c.level, c.replicate))
    if not ordered or ordered[0].level != 0:
        raise ValueError("L0 (clean) must be evaluated: it sets the threshold and denominator")

    # preprocess_fn(), never detector.preprocess: the dataset is forked into
    # DataLoader workers, and a bound method would drag the model with it.
    preprocess = detector.preprocess_fn()

    def run(condition):
        dataset = AIGCDataset(manifest, preprocess=preprocess, condition=condition)
        return score_dataset(dataset=dataset, detector=detector,
                             batch_size=batch_size,
                             num_workers=num_workers, device=device)

    clean_scores, labels, _ = run(ordered[0])
    threshold = M.threshold_from_clean(clean_scores, labels)
    clean_auc = M.roc_auc(clean_scores, labels)

    # pooled rows per level, for the level table and its CI
    pooled: dict[int, dict] = {
        0: {"scores": [clean_scores], "labels": [labels],
            "index": [np.array(manifest.index)], "recipes": []}
    }

    condition_rows: dict[str, dict] = {}
    for condition in ordered[1:]:
        scores, cond_labels, records = run(condition)
        bucket = pooled.setdefault(
            condition.level, {"scores": [], "labels": [], "index": [], "recipes": []}
        )
        bucket["scores"].append(scores)
        bucket["labels"].append(cond_labels)
        bucket["index"].append(np.array([r["index"] for r in records]))
        bucket["recipes"].extend(list(r["transforms"]) for r in records)

        if condition.level == 1:
            step = condition.steps[0]
            spec = TRANSFORMS[step.transform]
            auc = M.roc_auc(scores, cond_labels)
            condition_rows[condition.id] = {
                "transform": step.transform,
                "group": spec.group,
                "param_name": spec.param_name,
                "param": step.param,
                "auc": auc,
                "retention": M.retention(auc, clean_auc),
                "score_shift": M.score_shift(clean_scores, scores),
                "errors": M.error_breakdown(scores, cond_labels, threshold).as_dict(),
            }

    level_results, recipe_tables = {}, {}
    for level, bucket in pooled.items():
        scores = np.concatenate(bucket["scores"])
        level_labels = np.concatenate(bucket["labels"])
        auc = M.roc_auc(scores, level_labels)
        entry = {
            "auc": auc,
            "retention": M.retention(auc, clean_auc),
            "errors": M.error_breakdown(scores, level_labels, threshold).as_dict(),
        }
        if level > 0:
            groups = np.concatenate(bucket["index"])
            entry["auc_ci"] = list(
                M.bootstrap_ci(M.roc_auc, scores, level_labels,
                               groups=groups, n=N_BOOTSTRAP)
            )
        if level >= 2:
            entry["_recipes"] = bucket["recipes"]
            table = M.per_recipe_breakdown(scores, level_labels, bucket["recipes"], threshold)
            table["retention"] = table["auc"].map(lambda a: M.retention(a, clean_auc))
            recipe_tables[f"L{level}"] = table.to_dict("records")
        level_results[level] = entry

    levels = aggregate_by_level(condition_rows, level_results)
    by_transform = aggregate_by_transform(condition_rows)
    retention_by_level = {
        f"L{level}": levels[LEVEL_KEYS[level]]["retention"]
        for level in (1, 2, 3)
        if LEVEL_KEYS[level] in levels
    }
    all_recipes = [r for rows in recipe_tables.values() for r in rows]

    return {
        "run_id": None,   # filled by run_eval
        "detector": detector.name,
        "dataset": None,  # filled by run_eval
        "n_images": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "clean_threshold": threshold,
        "levels": levels,
        "conditions": condition_rows,
        "by_transform": by_transform,
        "recipes": recipe_tables,
        "summary": {
            "clean_auc": clean_auc,
            "retention_by_level": retention_by_level,
            "worst_condition": min(
                condition_rows, key=lambda k: condition_rows[k]["auc"], default=None
            ),
            "worst_recipe": min(
                all_recipes, key=lambda r: r["auc"], default={}
            ).get("transforms"),
            "operating_envelope": M.operating_envelope(retention_by_level, retention_floor),
        },
    }


def run_eval(cfg) -> list[dict]:
    """Top-level entry: one detector over every dataset in the run.

    The detector is built once and reused across datasets -- loading weights is
    the expensive part, and nothing about it is per-dataset. Each dataset gets
    its own clean threshold and its own retention denominator, so results stay
    independent and directly comparable.
    """
    grid = load_grid(cfg.degrade.grid_file, cfg.degrade.transforms)
    conditions = build_conditions(
        grid, cfg.degrade.levels, cfg.degrade.n_replicates, seed=cfg.degrade.seed
    )
    detector = build_detector(cfg.detector)

    results = []
    try:
        for dataset_cfg in cfg.datasets:
            manifest = sample_eval_subset(
                load_manifest(dataset_cfg.manifest, split=dataset_cfg.split),
                cfg.max_images,
                seed=cfg.degrade.seed,
            )
            result = evaluate_detector(
                detector, manifest, conditions, cfg.detector.device,
                batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                retention_floor=cfg.retention_floor,
            )
            result["run_id"] = cfg.run_id
            result["dataset"] = dataset_cfg.name
            write_json(
                result,
                Path(cfg.out_dir) / f"{cfg.run_id}__{detector.name}__{dataset_cfg.name}.json",
            )
            results.append(result)
    finally:
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
