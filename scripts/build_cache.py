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

from train.cache.schedule import EpochSchedule, val_epochs
from train.cache.spec import CacheSpec, sha_detector, sha_manifest, sha_preprocess
from train.cache.writer import build_cache
from train.config import load_cache_config
from eval.splits import build_split
from load_data.config import load_dataset_config
from eval.config import load_detector_config
from load_data.manifest import load_manifest, sample_eval_subset
from preprocessing.degrade.conditions import load_grid
from eval.detectors import build_detector, resolve_device


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

    detector_cfg = load_detector_config(cfg.detector)
    dataset_cfg = load_dataset_config(cfg.dataset)
    manifest = sample_eval_subset(
        load_manifest(dataset_cfg.manifest, dataset_cfg.split),
        cfg.max_images,
        seed=cfg.schedule.seed,
    )

    detector = build_detector(detector_cfg)
    split = build_split(detector, cfg.split, **cfg.split_args)
    schedule = EpochSchedule(
        grid=load_grid(cfg.schedule.grid_file, cfg.schedule.transforms),
        level_weights={int(k): v for k, v in cfg.schedule.level_weights.items()},
        seed=cfg.schedule.seed,
    )

    epochs = [*range(cfg.n_epochs), *val_epochs(cfg.n_val_epochs)]
    spec = CacheSpec(
        detector=detector_cfg.name,
        feature=split.feature_spec,
        n=len(manifest),
        shard_size=cfg.shard_size,
        manifest_sha=sha_manifest(manifest),
        schedule_sha=schedule.fingerprint(),
        detector_sha=sha_detector(detector_cfg),
        preprocess_sha=sha_preprocess(split.preprocess_fn()),
        # Empty unless `crop.enabled`, and empty is a claim -- "whole images" --
        # not a missing value. `preprocess_sha` cannot cover this: the window is
        # drawn in the dataset, before preprocessing, so that it can be seeded on
        # the image index without making the transform stochastic.
        crop_sha=cfg.crop.fingerprint(),
        # Empty unless `split_args.tap_blocks` was set, so a cache config that
        # says nothing about taps renders exactly what it always did.
        taps=split.taps(),
        tap_feature=split.tap_spec(),
        # Likewise None unless `freq.enabled`. Set and cleared together with
        # `freq_sha` -- `build_cache` refuses a spec that claims a view no
        # extractor is going to write.
        freq_feature=cfg.freq.feature(),
        freq_sha=cfg.freq.fingerprint(),
    )

    root = f"{cfg.out_dir.rstrip('/')}/{detector_cfg.name}"
    gb = spec.nbytes(len(epochs) + 1) / 1e9
    print(f"detector   {detector_cfg.name}")
    print(
        f"features   {spec.feature.layout}{spec.feature.shape} {spec.feature.dtype}"
        f"  ({spec.feature.bytes_per_image() / 1024:.1f} KB/image/view)"
    )
    if spec.taps:
        print(
            f"taps       {list(spec.taps)} {spec.tap_feature.shape} "
            f"({spec.tap_feature.bytes_per_image() / 1024:.1f} KB/image/view, "
            f"{spec.tap_feature.bytes_per_image() / spec.feature.bytes_per_image():.1f}x "
            f"the features)"
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
