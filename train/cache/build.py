"""Resolve a cache config into the inputs a render or append needs.

`scripts/main/build_cache.py` and `scripts/misc/append_cache.py` share this so the
`CacheSpec` -- and therefore every fingerprint it carries -- is computed
identically in both. A drift here would let an append silently reuse shards
rendered from different inputs, which is exactly the failure
`spec.assert_appendable` exists to refuse.
"""

from train.cache.schedule import EpochSchedule, val_epochs
from train.cache.spec import CacheSpec, sha_detector, sha_manifest, sha_preprocess
from eval.splits import build_split
from load_data.config import load_dataset_config
from eval.config import load_detector_config
from load_data.manifest import load_manifest, sample_eval_subset
from preprocessing.degrade.conditions import load_grid
from eval.detectors import build_detector


def resolve_cache_inputs(cfg):
    """Everything a cache build or append needs, resolved in one place.

    Returns `(detector_cfg, root, manifest, split, schedule, epochs, spec)`.
    `root` is derived from the detector name exactly as `build_cache.py` has
    always done, so a full render and an append land in the same directory.
    """
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
        # None unless `freq.enabled`. Set and cleared together with `freq_sha` --
        # the writer refuses a spec that claims a view no extractor will write.
        freq_feature=cfg.freq.feature(),
        freq_sha=cfg.freq.fingerprint(),
    )

    root = f"{cfg.out_dir.rstrip('/')}/{detector_cfg.name}"
    return detector_cfg, root, manifest, split, schedule, epochs, spec
