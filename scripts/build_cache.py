"""Render a feature cache. The expensive step, run once per detector.

    python scripts/build_cache.py train/configs/cache/dinov3.yaml --dry-run
    python scripts/build_cache.py train/configs/cache/dinov3.yaml

`--dry-run` prints the CacheSpec and the on-disk size and exits.

Resumable at shard granularity -- rerun after an interruption and it picks up at
the last checkpoint. Views already carrying `.done` from a completed earlier
render are skipped entirely, so adding epochs to a finished cache renders only
the new ones.
"""

import argparse

from train.cache.build import resolve_cache_inputs
from train.cache.writer import build_cache
from train.config import load_cache_config
from eval.detectors import resolve_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true", help="print the spec and size, then exit")
    p.add_argument("--epochs", type=int, help="override n_epochs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cache_config(args.config)
    if args.epochs is not None:
        cfg.n_epochs = args.epochs

    detector_cfg, root, manifest, split, schedule, epochs, spec = resolve_cache_inputs(cfg)

    gb = spec.nbytes(len(epochs) + 1) / 1e9
    print(f"detector   {detector_cfg.name}")
    print(
        f"features   {spec.feature.layout}{spec.feature.shape} {spec.feature.dtype}"
        f"  ({spec.feature.bytes_per_image() / 1024:.1f} KB/image/view)"
    )
    if spec.freq_feature is not None:
        print(
            f"freq       {spec.freq_feature.shape} {spec.freq_feature.dtype} "
            f"({spec.freq_feature.bytes_per_image() / 1024:.1f} KB/image/view, "
            f"{spec.freq_feature.bytes_per_image() / spec.feature.bytes_per_image():.1f}x "
            f"the features) -- {cfg.freq.patch}x{cfg.freq.patch} blocks, "
            f"{cfg.freq.grid}x{cfg.freq.grid} cells"
        )
    print(f"images     {spec.n}")
    print(
        f"window     {cfg.crop.s_min}-{cfg.crop.s_max}px {cfg.crop.policy}, "
        f"one per image"
        if cfg.crop.enabled
        else "window     whole images (crop disabled)"
    )
    print(f"views      {len(epochs) + 1}  (clean + {cfg.n_epochs} train + {cfg.n_val_epochs} val)")
    print(f"total      {gb:.1f} GB -> {root}")
    if args.dry_run:
        return

    build_cache(
        split, manifest, root, spec, schedule, epochs,
        batch_size=cfg.batch_size, trunk_batch_size=cfg.trunk_batch_size,
        num_workers=cfg.num_workers, device=resolve_device(cfg.device),
        crop=cfg.crop.build(), freq=cfg.freq.build(),
    )
    print(f"done: {root}")


if __name__ == "__main__":
    main()
