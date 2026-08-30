"""YAML <-> dataclass config loading for what a dataset IS.

    configs/datasets/<name>.yaml    how to build a dataset, where its manifest
                                    lives, which split to score

`source` is only needed by scripts/build_manifest.py; once the manifest
exists, everything downstream reads `manifest` and `split` and ignores it.

Components are named by dotted import path rather than by a registry key, so
pointing a dataset config at a new source never requires editing this module.
See `common.imports`.

Paths inside a config are resolved relative to the current working directory,
which every script in this project assumes is the repo root.
"""

from dataclasses import dataclass

from common.io import resolve_ref


@dataclass
class DatasetConfig:
    """One dataset: how to materialize it, and how to read it back.

    `source` is only needed by scripts/build_manifest.py; once the manifest
    exists, evaluation reads `manifest` and `split` and ignores it.
    """

    name: str                     # short id, used in result filenames
    manifest: str                 # path to manifest.parquet
    split: str | None = None      # which split to evaluate on; None = every row
    source: dict | None = None    # {target, args} spec for build_manifest


def load_dataset_config(entry: str | dict) -> DatasetConfig:
    """Load one dataset spec, as used by both build_manifest and run_eval."""
    entry = resolve_ref(entry)
    return DatasetConfig(
        name=entry["name"],
        manifest=entry["manifest"],
        split=entry.get("split"),
        source=entry.get("source"),
    )
