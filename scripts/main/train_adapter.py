"""Stage 1: train one label-free adapter against one rendered cache.

    python scripts/main/train_adapter.py train/configs/train/dinov3_multiscale.yaml
    python scripts/main/train_adapter.py train/configs/train/dinov3_multiscale.yaml --seed 1 --run-id dinov3_multiscale_s1

CLI overrides exist for one reason: the seed and geometry sweep is this script in
a shell loop, and a sweep should not need a config file per point.

Evaluation is not here. A trained checkpoint is scored by pointing the eval
harness at `eval/configs/detectors/<base>+grace.yaml`, so adapted and baseline numbers
come out of the same code path.
"""

import argparse

from train.cache.schedule import EpochSchedule
from train.config import load_train_config
from eval.splits import build_split
from train.tracker import add_wandb_args, apply_wandb_args
from train.loop import train_adapter
from load_data.config import load_dataset_config
from eval.config import load_detector_config
from load_data.manifest import load_manifest
from preprocessing.degrade.conditions import load_grid
from eval.detectors import build_detector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int, help="cap the training epochs read from the cache")
    p.add_argument("--run-id")
    p.add_argument("--out-dir")
    p.add_argument("--bottleneck", type=int)
    p.add_argument("--n-blocks", type=int)
    add_wandb_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_train_config(args.config)
    for key in ("seed", "epochs", "run_id", "out_dir"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    for key in ("bottleneck", "n_blocks"):
        if getattr(args, key) is not None:
            setattr(cfg.adapter, key, getattr(args, key))
    apply_wandb_args(cfg, args)

    dataset_cfg = load_dataset_config(cfg.dataset)
    manifest = load_manifest(dataset_cfg.manifest, dataset_cfg.split)
    split = build_split(
        build_detector(load_detector_config(cfg.detector)), cfg.split, **cfg.split_args
    )
    schedule = EpochSchedule(
        grid=load_grid(cfg.schedule.grid_file, cfg.schedule.transforms),
        level_weights={int(k): v for k, v in cfg.schedule.level_weights.items()},
        seed=cfg.schedule.seed,
    )

    summary = train_adapter(cfg, split, manifest, schedule)
    print(f"{cfg.run_id}: {summary['steps']} steps")
    for name, row in (summary["validation"] or {}).items():
        print(f"  {name}: {row}")


if __name__ == "__main__":
    main()
