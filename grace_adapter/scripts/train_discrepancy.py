"""Stage 2: train the discrepancy head against a FROZEN stage-1 adapter.

    python scripts/train_discrepancy.py configs/train/rine_discrepancy.yaml
    python scripts/train_discrepancy.py configs/train/rine_discrepancy.yaml --adapter checkpoints/grace/rine_clean/step_001000.pt --run-id e4_step1000

This is the only part of GRACE that uses image labels, which is why it is a
separate script against a frozen adapter: GRACE and GRACE-D then ship the same
adapter weights, bit for bit, and the label-free claim stays literally true of
the shipped adapter.

Seconds per run, which is what makes experiment E4 practical -- point `--adapter`
at each intermediate stage-1 checkpoint in turn and compare `auc_aux` across
them. A falling `auc_aux` as stage 1 improves is direct evidence that restoring
features destroys the forensic drift signal, and the resulting
retention-versus-drift-preservation curve is the figure.
"""

import argparse

from grace.config import load_discrepancy_config
from grace.splits import build_split
from grace.train.tracker import add_wandb_args, apply_wandb_args
from grace.train.loop import train_discrepancy
from pipeline.config import load_dataset_config, load_detector_config
from pipeline.data.manifest import load_manifest
from pipeline.detectors import build_detector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--adapter", help="override adapter_checkpoint (experiment E4)")
    p.add_argument("--run-id")
    p.add_argument("--dataset", help="override the dataset config path")
    add_wandb_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_discrepancy_config(args.config)
    if args.adapter:
        cfg.adapter_checkpoint = args.adapter
    if args.run_id:
        cfg.run_id = args.run_id
    apply_wandb_args(cfg, args)

    dataset_cfg = load_dataset_config(args.dataset or cfg.dataset)
    manifest = load_manifest(dataset_cfg.manifest, dataset_cfg.split)
    split = build_split(
        build_detector(load_detector_config(cfg.detector)), cfg.split, **cfg.split_args
    )

    summary = train_discrepancy(cfg, split, manifest)
    print(f"{cfg.run_id}: beta={summary['beta']:+.4f}")
    for axis, rows in summary["validation"].items():
        print(f"  {axis}:")
        for name, row in rows.items():
            print(
                f"    {name}: main={row['auc_main']:.4f}  aux={row['auc_aux']:.4f}  "
                f"fused={row['auc_fused']:.4f}  (fused-main={row['auc_fused']-row['auc_main']:+.4f})"
            )


if __name__ == "__main__":
    main()
