"""YAML <-> dataclass config loading.

Three kinds of config, one file each, one directory each:

    configs/datasets/<name>.yaml    what a dataset IS -- how to build it, where
                                    its manifest lives, which split to score
    configs/detectors/<name>.yaml   what a detector IS -- import path and args
    configs/runs/<name>.yaml        one detector x a set of datasets, plus how
                                    this particular run executes

`configs/defaults.yaml` is the annotated reference: every key with its default,
not something the harness loads.

Adding a dataset is one new file under datasets/. Adding a detector is one new
file under detectors/. Pairing them is one new file under runs/. Nothing is
restated: a run references the other two by path.

Components are named by dotted import path rather than by a registry key, so
pointing the harness at a new model or dataset never requires editing pipeline
code. See `pipeline.utils.imports`.

Paths inside a config are resolved relative to the current working directory,
which the README's usage assumes is the repo root.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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


@dataclass
class DetectorConfig:
    name: str                          # display name, used in result filenames
    target: str                        # dotted import path to a FrozenDetector
    args: dict = field(default_factory=dict)
    device: str = "auto"               # "auto" | "cpu" | "cuda" | "cuda:1" | "mps"


@dataclass
class DegradeConfig:
    """Every field defaults, so `degrade: {}` gives the full standard sweep."""

    grid_file: str = "configs/degradations.yaml"
    levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    n_replicates: int = 3             # independent re-draws of L2/L3
    transforms: list[str] | None = None   # None = all six
    seed: int = 0


@dataclass
class RunConfig:
    """One detector scored against one or more datasets.

    Loader knobs live here rather than on the dataset: `batch_size` and
    `num_workers` are properties of the machine this run is on, and
    `max_images` is a scope decision for this run -- none of them describe what
    the dataset is.

    Only `run_id`, `detector` and `datasets` have no default.
    """

    run_id: str
    detector: DetectorConfig
    datasets: list[DatasetConfig]
    out_dir: str = "results/"
    max_images: int | None = None     # per class; None = all
    batch_size: int = 32
    num_workers: int = 4              # raise on a cluster node, lower on a laptop
    degrade: DegradeConfig = field(default_factory=DegradeConfig)
    retention_floor: float = 0.5      # see eval.metrics.RETENTION_FLOOR


def read_yaml(path: str | Path) -> Any:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def _resolve(entry: str | dict) -> dict:
    """A reference is either a path to a spec file or the same mapping inline."""
    return read_yaml(entry) if isinstance(entry, (str, Path)) else entry


def load_detector_config(entry: str | dict) -> DetectorConfig:
    """Load one detector spec.

    The indirection lets run_eval.py and predict.py share one definition of a
    detector instead of restating its target and args in two places.
    """
    entry = _resolve(entry)
    return DetectorConfig(
        name=entry["name"],
        target=entry["target"],
        args=entry.get("args") or {},
        device=entry.get("device", "auto"),
    )


def load_dataset_config(entry: str | dict) -> DatasetConfig:
    """Load one dataset spec, as used by both build_manifest and run_eval."""
    entry = _resolve(entry)
    return DatasetConfig(
        name=entry["name"],
        manifest=entry["manifest"],
        split=entry.get("split"),
        source=entry.get("source"),
    )


def load_run_config(path: str | Path) -> RunConfig:
    """Parse a run yaml into a RunConfig.

    Absent keys fall through to the dataclass defaults rather than being
    restated here, so there is exactly one place a default is written down.
    """
    raw = read_yaml(path)
    optional = {
        k: raw[k]
        for k in ("out_dir", "max_images", "batch_size", "num_workers", "retention_floor")
        if raw.get(k) is not None
    }
    return RunConfig(
        run_id=raw["run_id"],
        detector=load_detector_config(raw["detector"]),
        datasets=[load_dataset_config(d) for d in raw["datasets"]],
        degrade=DegradeConfig(**(raw.get("degrade") or {})),
        **optional,
    )
