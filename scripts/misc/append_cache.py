"""Append new manifest rows to an already-rendered feature cache.

    python scripts/misc/append_cache.py train/configs/cache/<combined>.yaml --dry-run
    python scripts/misc/append_cache.py train/configs/cache/<combined>.yaml

The cache config names the COMBINED manifest -- the old rows first, the new rows
appended after them -- and the SAME detector, schedule, crop and freq blocks the
existing cache was rendered under. The existing cache must already sit at the
detector's `out_dir` root, rendered from the OLD manifest. This script reuses its
shards byte-for-byte and renders only the new rows, appending them as additional
shards.

It refuses to run unless the old and new inputs agree on every fingerprint the
features depend on (detector, schedule, preprocess, crop, freq) and unless the
old manifest is an exact prefix of the new one. That check is the whole point:
reusing shards across a fingerprint change would produce a plausible-looking
cache whose first rows are features of different inputs than its last rows, and
no reader would notice.

One-way: `index.npy` and `spec.json` are overwritten with the grown versions, so
the cache can no longer be read against the old manifest. Copy the old cache
directory first if the pre-append numbers must stay reproducible.
"""

import argparse

from pathlib import Path

from train.cache.spec import CacheSpec, assert_appendable
from train.cache.writer import append_cache
from train.cache.build import resolve_cache_inputs
from train.config import load_cache_config
from eval.detectors import resolve_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true", help="print the plan, then exit")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cache_config(args.config)

    detector_cfg, root, manifest, split, schedule, epochs, spec = resolve_cache_inputs(cfg)
    root = Path(root)

    try:
        old_spec = CacheSpec.load(root)
    except FileNotFoundError as e:
        raise SystemExit(
            f"{root} has no cache to append to. Render it first with "
            f"scripts/main/build_cache.py."
        ) from e

    old_n = assert_appendable(old_spec, spec, manifest, epochs)

    print(f"detector   {detector_cfg.name}")
    print(f"cache      {root}")
    print(f"existing   {old_spec.n} rows")
    print(f"new        {spec.n} rows  (+{spec.n - old_spec.n})")
    print(f"reuse      {old_n} rows (copied, never re-computed)")
    print(f"render     {spec.n - old_n} rows -> {spec.n - old_spec.n} net new")
    if args.dry_run:
        return

    append_cache(
        split, manifest, root, spec, schedule, epochs, old_n,
        batch_size=cfg.batch_size, trunk_batch_size=cfg.trunk_batch_size,
        num_workers=cfg.num_workers, device=resolve_device(cfg.device),
        crop=cfg.crop.build(), freq=cfg.freq.build(),
    )
    print(f"done: {root} ({spec.n} rows)")


if __name__ == "__main__":
    main()
