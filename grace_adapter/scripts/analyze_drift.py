"""Test RA-Det's premise on this data, before anything is TRAINED on it.

    python scripts/build_cache.py configs/cache/rine.yaml        # PREREQUISITE
    python scripts/analyze_drift.py --cache cache/rine --dataset ../eval_pipeline/configs/datasets/sid_set.yaml

**Needs a rendered cache**, including at least one finalized degraded view: the
whole analysis is a comparison of `clean` against `epoch=NNN`, so there is
nothing to compute before `build_cache.py` has run. `FeatureCache` is opened in
the first statement of `main`, so a missing cache fails immediately, and a cache
with no finalized degraded view exits with `no rendered epochs under <dir>`.

Given that cache it reads nothing else: no training, no GPU, minutes. Reports,
per level and per transform, how far generated images drift under degradation
versus real ones, and how much of that drift lies inside the frozen head's
sensitive subspace.

Both outcomes are useful, which is why this runs before stage 1:

  * **Asymmetry present** -> drift carries forensic signal, the discrepancy branch
    has something to read, and the label-free objective is knowingly erasing it.
    Proceed with stage 2 and expect the trade-off curve from experiment E4.
  * **Asymmetry absent** -> the discrepancy branch will be weak here. Say so,
    keep the restoration result, and save a day. That is a finding about this
    dataset, not a refutation of RA-Det.

The parallel/orthogonal split matters as much as the gap itself. Drift that is
large but orthogonal to the decision direction is invisible to the frozen head --
which is exactly why an auxiliary head reading Δ can recover signal the main head
cannot, and why the main head's loss can fall while evidence is destroyed.

`--split` and `--detector` are optional: without them the decision-direction
decomposition is skipped and only the magnitude gap is reported, which needs no
model weights at all.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from grace.cache.reader import FeatureCache
from grace.train import diagnostics as D
from pipeline.config import load_dataset_config, load_detector_config
from pipeline.data.manifest import load_manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="a rendered cache directory")
    p.add_argument("--dataset", required=True, help="the harness dataset config")
    p.add_argument("--detector", help="harness detector config; enables the ∥/⊥ split")
    p.add_argument("--split", help="dotted path to the SplitDetector")
    p.add_argument("--epochs", type=int, nargs="*", help="default: every rendered epoch")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", default="results/drift.json")
    return p.parse_args()


def load_split(args):
    """The split, built ONCE, or None if the ∥/⊥ decomposition was not requested.

    Hoisted out of the batch loop deliberately. `build_detector` loads weights
    from disk and runs `verify_split`, so building it per batch reloads the whole
    trunk for every 512 images -- dozens of times over a multi-epoch cache, for a
    gradient that depends only on the frozen head.
    """
    if not (args.detector and args.split):
        return None
    from grace.splits import build_split
    from pipeline.detectors import build_detector

    return build_split(build_detector(load_detector_config(args.detector)), args.split)


def head_gradient_for(split, f_clean):
    """∇_f head(f) at the clean features, returned on `f_clean`'s device.

    The cache hands back CPU tensors -- it is memmapped numpy -- while the
    detector sits whereever its config's `device:` put it, which for `auto` is
    MPS or CUDA. So the features go to the model and the gradient comes back, and
    every other quantity in this script stays on the CPU where the cache put it.
    """
    if split is None:
        return None
    from grace.train.weighting import head_gradient

    param = next(split.parameters(), None)
    device = param.device if param is not None else f_clean.device
    return head_gradient(split.head, f_clean.to(device)).to(f_clean.device)


def main():
    args = parse_args()
    cache = FeatureCache(args.cache)
    dataset = load_dataset_config(args.dataset)
    manifest = load_manifest(dataset.manifest, dataset.split)

    index = np.asarray(manifest.index, dtype=np.int64)
    # `np.array`, not `np.asarray`: a pandas column can hand back a read-only
    # view, and torch warns about wrapping one on every run.
    labels = torch.from_numpy(np.array(manifest["label"], dtype=np.int64))
    epochs = args.epochs or list(cache.epochs())
    if not epochs:
        raise SystemExit(f"no rendered epochs under {args.cache}")

    # Once, before the loop -- see load_split.
    split = load_split(args)

    report = {"cache": str(args.cache), "dataset": dataset.name, "epochs": {}}
    for epoch in epochs:
        recipes = cache.recipes(epoch)
        per_batch, drift_all, level_all, transform_all = [], [], [], []

        for start in range(0, len(index), args.batch_size):
            sel = index[start : start + args.batch_size]
            f_clean = cache.clean(sel).float()
            f_deg = cache.degraded(sel, epoch).float()
            y = labels[start : start + args.batch_size]

            j = head_gradient_for(split, f_clean)
            per_batch.append(D.drift_asymmetry(f_deg, f_clean, y, j))
            drift_all.append(D.drift(f_deg, f_clean)["relative"].numpy())
            level_all.append(recipes.loc[sel, "level"].to_numpy())
            transform_all.extend(recipes.loc[sel, "transforms"])

        drift = np.concatenate(drift_all)
        y = labels.numpy().astype(bool)
        lo, hi = D.bootstrap_gap(drift, y)

        entry = {
            "overall": _mean_of(per_batch),
            "asymmetry_ci": [lo, hi],
            # A gap whose CI straddles zero is not evidence, however large the
            # point estimate. This is the number that decides whether stage 2 is
            # worth building.
            "significant": bool(lo > 0 or hi < 0),
            "by_level": _group(drift, y, np.concatenate(level_all)),
            "by_transform": _by_transform(drift, y, transform_all),
        }
        report["epochs"][f"epoch_{epoch}"] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print(report)


def _mean_of(dicts: list[dict]) -> dict:
    keys = {k for d in dicts for k in d}
    return {
        k: float(np.mean([d[k] for d in dicts if k in d]))
        for k in keys
        if not k.startswith("n_")
    }


def _group(drift: np.ndarray, labels: np.ndarray, key: np.ndarray) -> dict:
    out = {}
    for value in np.unique(key):
        sel = key == value
        if labels[sel].any() and (~labels[sel]).any():
            out[str(value)] = {
                "drift_real": float(drift[sel & ~labels].mean()),
                "drift_fake": float(drift[sel & labels].mean()),
                "asymmetry": float(drift[sel & labels].mean() - drift[sel & ~labels].mean()),
                "n": int(sel.sum()),
            }
    return out


def _by_transform(drift: np.ndarray, labels: np.ndarray, transforms: list) -> dict:
    """One row per transform that appeared, over every recipe containing it.

    Not disjoint -- a composed recipe contributes to each of its transforms --
    which is the same convention the harness's `by_transform` table uses.
    """
    out = {}
    for name in sorted({t for row in transforms for t in row}):
        mask = np.array([name in row for row in transforms])
        real, fake = drift[mask & ~labels], drift[mask & labels]
        if len(real) and len(fake):
            out[name] = {
                "drift_real": float(real.mean()),
                "drift_fake": float(fake.mean()),
                "asymmetry": float(fake.mean() - real.mean()),
                "n": int(mask.sum()),
            }
    return out


def _print(report: dict) -> None:
    for epoch, entry in report["epochs"].items():
        overall = entry["overall"]
        print(f"\n{epoch}")
        print(f"  drift  real={overall.get('drift_real', float('nan')):.4f}  "
              f"fake={overall.get('drift_fake', float('nan')):.4f}")
        print(f"  asymmetry={overall.get('asymmetry', float('nan')):+.4f}  "
              f"CI={entry['asymmetry_ci']}  "
              f"{'SIGNIFICANT' if entry['significant'] else 'not significant'}")
        if "parallel_fraction" in overall:
            print(f"  fraction of drift inside the decision subspace: "
                  f"{overall['parallel_fraction']:.3f}")


if __name__ == "__main__":
    main()
