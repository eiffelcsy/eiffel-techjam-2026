"""YAML <-> dataclass config loading for what a detector IS and what a run does.

    configs/detectors/<name>.yaml   what a detector IS -- import path and args
    configs/runs/<name>.yaml        a set of detectors x a set of datasets, plus
                                    how this particular run executes

`configs/defaults.yaml` is the annotated reference: every key with its default,
not something the harness loads.

Adding a detector is one new file under detectors/. Pairing it with datasets is
one new file under runs/. Nothing is restated: a run references a dataset
config (`load_data.config.DatasetConfig`) by path.

Components are named by dotted import path rather than by a registry key, so
pointing the harness at a new model never requires editing eval code. See
`common.imports`.

Paths inside a config are resolved relative to the current working directory,
which every script in this project assumes is the repo root.
"""

from dataclasses import dataclass, field
from pathlib import Path

from common.io import read_yaml, resolve_ref
from load_data.config import DatasetConfig, load_dataset_config


@dataclass
class DetectorConfig:
    name: str                          # display name, used in result filenames
    target: str                        # dotted import path to a FrozenDetector
    args: dict = field(default_factory=dict)
    device: str = "auto"               # "auto" | "cpu" | "cuda" | "cuda:1" | "mps"


@dataclass
class DegradeConfig:
    """Every field defaults, so `degrade: {}` gives the full standard sweep."""

    grid_file: str = "preprocessing/configs/degradations.yaml"
    levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    n_replicates: int = 3             # independent re-draws of L2/L3
    transforms: list[str] | None = None   # None = all eleven
    seed: int = 0


@dataclass
class RunConfig:
    """A set of detectors scored against a set of datasets.

    Loader knobs live here rather than on the dataset: `batch_size` and
    `num_workers` are properties of the machine this run is on, and
    `max_images` is a scope decision for this run -- none of them describe what
    the dataset is.

    Detectors are a list so several are one command and one results directory.
    Each is loaded, scored across every dataset, and released before the next
    is built, so a run costs one detector's memory however many are named.
    Note that the loader knobs are shared: a detector needing its own
    `batch_size` -- one taking images at native resolution, which cannot be
    stacked -- belongs in its own run rather than in a shared list.

    Only `run_id`, `detectors` and `datasets` have no default.
    """

    run_id: str
    detectors: list[DetectorConfig]
    datasets: list[DatasetConfig]
    out_dir: str = "results/"
    max_images: int | None = None     # per class; None = all
    batch_size: int = 32
    num_workers: int = 4              # raise on a cluster node, lower on a laptop
    degrade: DegradeConfig = field(default_factory=DegradeConfig)
    retention_floor: float = 0.5      # see eval.metrics.RETENTION_FLOOR


def load_detector_config(entry: str | dict) -> DetectorConfig:
    """Load one detector spec.

    The indirection lets run_eval.py and predict.py share one definition of a
    detector instead of restating its target and args in two places.
    """
    entry = resolve_ref(entry)
    return DetectorConfig(
        name=entry["name"],
        target=entry["target"],
        args=entry.get("args") or {},
        device=entry.get("device", "auto"),
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

    # `detector:` and `detectors:` are the same key at different arities. One
    # detector is the common case and reads better singular; a zoo needs the
    # list. Accepting both costs three lines and saves every one-detector run
    # config from a one-element list.
    if ("detector" in raw) == ("detectors" in raw):
        raise KeyError("a run config needs exactly one of `detector:` or `detectors:`")
    entries = raw["detectors"] if "detectors" in raw else [raw["detector"]]

    return RunConfig(
        run_id=raw["run_id"],
        detectors=[load_detector_config(d) for d in entries],
        datasets=[load_dataset_config(d) for d in raw["datasets"]],
        degrade=DegradeConfig(**(raw.get("degrade") or {})),
        **optional,
    )
