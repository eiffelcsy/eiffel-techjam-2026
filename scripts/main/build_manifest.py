"""Materialize a dataset and build its manifest.

    python scripts/main/build_manifest.py --config load_data/configs/datasets/<name>.yaml

The dataset config is the same file run_eval.py reads: `source` says how to
build it, `manifest` says where the table goes. See load_data/configs (dataset shape) and eval/configs/defaults.yaml.
"""

import argparse

from load_data.config import load_dataset_config
from load_data.manifest import build_manifest
from common.imports import instantiate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="dataset spec yaml")
    p.add_argument("--out", default=None, help="override the spec's `manifest` path")
    args = p.parse_args()

    cfg = load_dataset_config(args.config)
    if cfg.source is None:
        raise SystemExit(
            f"{args.config} has no `source:` block, so there is nothing to build. "
            "Add one, or point --config at a dataset spec that has it."
        )

    out = args.out or cfg.manifest
    df = build_manifest(instantiate(cfg.source), out)

    counts = df["label"].value_counts()
    print(f"wrote {out}: {len(df)} rows "
          f"({counts.get(0, 0)} real / {counts.get(1, 0)} generated)")
    print(df.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
