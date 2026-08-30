"""Stage 0: fit the PoC detector's classification head on clean features.

    python scripts/train_probe.py configs/probe/dinov3_wildfake.yaml

Run this once, before build_cache.py. Nothing else in the pipeline trains a
detector; this exists because the PoC detector is a DINOv3 trunk plus a head
that has to come from somewhere. See `grace.probe`.

The head is written to the path the *detector config* names in
`args.head_checkpoint`, so the file the probe produces and the file the detector
loads are the same string in one place. `--out` overrides that for a variant.

Ordering note: this script builds the detector with `head_checkpoint: null` --
the head does not exist yet, and only the trunk is needed to extract features.
The split warns loudly about the untrained head and is right to; here it is
expected.
"""

import argparse
import json
from pathlib import Path

from grace.config import load_probe_config
from grace.probe import train_probe
from grace.splits import build_split
from grace.train.tracker import add_wandb_args, apply_wandb_args
from load_data.config import load_dataset_config
from eval.config import load_detector_config
from load_data.manifest import load_manifest
from eval.detectors import build_detector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--out", help="override the head checkpoint path")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--run-id")
    add_wandb_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_probe_config(args.config)
    for key in ("out", "epochs", "seed", "run_id"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    apply_wandb_args(cfg, args)

    detector_cfg = load_detector_config(cfg.detector)
    if not cfg.out:
        cfg.out = detector_cfg.args.get("head_checkpoint") or ""
    if not cfg.out:
        raise SystemExit(
            f"{cfg.detector} names no `head_checkpoint`, and the probe config sets "
            f"no `out`. One of them must say where the head goes."
        )

    # The head is what this script produces; loading one here would train a
    # second head on top of an existing one and quietly report its AUC.
    detector_cfg.args = {**detector_cfg.args, "head_checkpoint": None}
    split = build_split(build_detector(detector_cfg), cfg.split)

    dataset_cfg = load_dataset_config(cfg.dataset)
    manifest = load_manifest(dataset_cfg.manifest, dataset_cfg.split)

    # Overlap is checked on `path`, not on the manifest index. The index is the
    # row number within ONE manifest file, so two separately-built manifests both
    # start at 0 and an index test reports a collision between datasets that
    # share no image at all -- which is exactly the case now that selection runs
    # against the manifest's own held-out `validation` split.
    # `path` is the image identity and is comparable across manifests.
    train_paths = set(manifest["path"])
    val_sets = []
    for path in cfg.val_datasets():
        val_cfg = load_dataset_config(path)
        val_manifest = load_manifest(val_cfg.manifest, val_cfg.split)
        shared = train_paths & set(val_manifest["path"])
        if shared:
            raise SystemExit(
                f"{cfg.dataset} and {path} share {len(shared)} image(s), e.g. "
                f"{sorted(shared)[0]}. The probe would be selected on images it "
                f"was fit on, and every downstream retention number would "
                f"inherit that."
            )
        val_sets.append((val_cfg.name, val_manifest))

    summary = train_probe(cfg, split, manifest, val_sets)
    counts = " / ".join(f"{n} {name}" for name, n in summary["n_val"].items())
    print(f"{cfg.run_id}: {summary['n_train']} train images | val: {counts or 'none'}")
    print(f"  selection   {summary['selection']} @ epoch {summary['selected_epoch']}")
    print(f"  {json.dumps(summary['history'][-1])}")
    print(f"  wrote       {summary['checkpoint']}")


if __name__ == "__main__":
    main()
