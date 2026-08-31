"""Build a combined manifest: the existing WildFake manifest + new sources.

    python scripts/main/build_combined_manifest.py load_data/configs/datasets/wildfake_train_combined.yaml

Unlike `scripts/main/build_manifest.py`, this does NOT rebuild from scratch. It loads
the existing manifest in place -- preserving its row order and indices, which
are the image identity that seeds every degradation and crop -- and appends the
new single-class sources with fresh, non-colliding indices. That is what makes
the feature-cache append safe: the first rows of the combined manifest are
byte-identical to the existing manifest, so `scripts/append_cache.py` can reuse
the already-rendered shards.

The build spec lives in the dataset config's `build:` block:

    build:
      base_manifest: data/wildfake_train/manifest.parquet
      out_manifest:  data/wildfake_plus_sota/manifest.parquet
      reals_root:    data/wildfake_train/images   # to map manifest paths back to CSV rel paths
      sources:       [ {target, args, ...}, ... ]

Each source's images land under `out_manifest`'s `images/partNN/`, so two sources
can never overwrite each other. A source whose spec carries `exclude_from_base:
true` (the added-reals source) is handed the existing manifest's real paths as
`exclude_paths`, so the new reals are disjoint from the ones already sampled.
"""

import argparse
from pathlib import Path

import pandas as pd

from common.imports import instantiate
from common.io import read_yaml
from load_data.manifest import COLUMNS, load_manifest, manifest_rel_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true", help="print the plan, then exit")
    args = p.parse_args()

    cfg = read_yaml(args.config)
    build = cfg.get("build") or {}
    if not build:
        raise SystemExit(
            f"{args.config} has no `build:` block -- there is nothing to combine."
        )

    out_manifest = Path(build["out_manifest"])
    base = load_manifest(build["base_manifest"])

    reals_root = Path(build.get("reals_root", "")) if build.get("reals_root") else None
    exclude_rel = (
        manifest_rel_paths(base, reals_root, 0) if reals_root is not None else []
    )

    rows = []
    for i, spec in enumerate(build["sources"]):
        overrides = {}
        if spec.get("exclude_from_base"):
            overrides["exclude_paths"] = exclude_rel
        source = instantiate(spec, **overrides)
        target = out_manifest.parent / "images" / f"part{i:02d}"
        for row in source.rows(target):
            rows.append(row)

    new_df = pd.DataFrame(rows, columns=COLUMNS)
    combined = pd.concat([base, new_df], ignore_index=True)

    print(f"base       {len(base)} rows (indices preserved)")
    print(f"new        {len(new_df)} rows")
    print(combined.groupby(["split", "label"]).size().to_string())

    if args.dry_run:
        return

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_manifest, index=False)
    print(f"wrote      {out_manifest} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
