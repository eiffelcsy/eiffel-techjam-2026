"""Stage 0: fit the PoC detector's classification head on clean features.

    python scripts/train_probe.py configs/probe/dinov3_ntire.yaml

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
from pipeline.config import load_dataset_config, load_detector_config
from pipeline.data.manifest import load_manifest
from pipeline.detectors import build_detector


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
    val_manifest = None
    if cfg.val_dataset:
        val_cfg = load_dataset_config(cfg.val_dataset)
        val_manifest = load_manifest(val_cfg.manifest, val_cfg.split)
        if set(manifest.index) & set(val_manifest.index):
            raise SystemExit(
                f"{cfg.dataset} and {cfg.val_dataset} share manifest rows. The probe "
                f"would be selected on images it was fit on, and every downstream "
                f"retention number would inherit that."
            )

    summary = train_probe(cfg, split, manifest, val_manifest)
    print(f"{cfg.run_id}: {summary['n_train']} train / {summary['n_val']} val images")
    print(f"  selection   {summary['selection']} @ epoch {summary['selected_epoch']}")
    print(f"  {json.dumps(summary['history'][-1])}")
    print(f"  wrote       {summary['checkpoint']}")


if __name__ == "__main__":
    main()
