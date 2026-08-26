"""YAML <-> dataclass config, mirroring `pipeline.config`.

Three kinds of config, one directory each:

    configs/cache/<name>.yaml       what to render, for which detector
    configs/train/<name>.yaml       one stage-1 or stage-2 run against a cache
    configs/detectors/<name>.yaml   the ADAPTED detector, in the harness's shape

Detectors and datasets are never redefined here: a cache config references the
harness's own `configs/detectors/*.yaml` and `configs/datasets/*.yaml` by path,
so each is described in exactly one place in the project and GRACE cannot drift
from what was benchmarked.

Absent keys fall through to the dataclass defaults rather than being restated, so
there is one place a default is written down. `configs/defaults.yaml` is the
annotated reference and is never loaded.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from grace.cache.schedule import DEFAULT_LEVEL_WEIGHTS


@dataclass
class ScheduleConfig:
    """The degradation distribution seen during training.

    Deliberately *not* the eval sweep. Eval is a controlled one-factor-at-a-time
    grid plus Monte-Carlo composition; training samples from the same operator
    pool under `level_weights`. Sharing `grid_file` with the harness keeps them
    the same eleven transforms at the same parameter values, which is the honest
    setting to report -- and makes "held-out transforms" a deliberate experiment
    (`transforms:`) rather than an accident.
    """

    grid_file: str = "../eval_pipeline/configs/degradations.yaml"
    transforms: list[str] | None = None
    level_weights: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_LEVEL_WEIGHTS)
    )
    seed: int = 0


@dataclass
class WandbConfig:
    """Weights & Biases, off by default and never load-bearing.

    `summary.json` next to the checkpoints stays the record of every run; this
    only mirrors it somewhere a sweep can be sorted. A run with tracking off, a
    run whose network died mid-way and a run whose dashboard was deleted are all
    equally reportable. See `grace.train.tracker`.

    `group` is what makes the sweeps in README section 7 legible: set it to the
    experiment id (`e3_losses`, `e4_erasure`) and every arm lands in one
    comparison rather than in a flat list of forty runs.

    `mode: offline` writes to `./wandb/` and syncs later, which is the setting
    for a cluster node with no outbound network.
    """

    enabled: bool = False
    project: str = "grace-adapter"
    entity: str | None = None
    group: str = ""
    tags: list[str] = field(default_factory=list)
    mode: str = "online"            # "online" | "offline" | "disabled"


@dataclass
class ProbeConfig:
    """Stage 0 -- the PoC only. Fit the frozen detector's own classification head.

    Exists because the PoC detector is assembled here rather than downloaded: a
    DINOv3 trunk has no classifier, and GRACE cannot adapt a seam whose head does
    not exist yet. Every other detector in the project arrives with its head
    already trained by its authors, and for those this config is unused.

    `out` defaults to whatever `head_checkpoint` the detector config names, so
    the path the probe writes and the path the detector loads are one string in
    one file. Override only to train a variant without editing the detector.
    """

    run_id: str
    detector: str
    dataset: str
    val_dataset: str = ""
    out: str = ""
    hidden: int = 512
    n_layers: int = 2
    dropout: float = 0.0
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    """Images per trunk forward during feature extraction."""
    head_batch_size: int = 64
    """Features per optimizer step. Unrelated to the above: the trunk runs once."""
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    split: str = "grace.splits.dinov3.DINOv3Split"
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class CacheConfig:
    """One render: the clean view plus `n_epochs` degraded views."""

    detector: str
    split: str
    dataset: str
    out_dir: str = "cache/"
    n_epochs: int = 12
    n_val_epochs: int = 2
    dtype: str = "float16"
    shard_size: int = 50_000
    batch_size: int = 32
    num_workers: int = 8
    max_images: int | None = None
    device: str = "auto"
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


@dataclass
class AdapterConfig:
    bottleneck: int = 256
    n_blocks: int = 2
    per_channel_gate: bool = True
    dropout: float = 0.0
    noise_dim: int = 0
    """0 disables posterior sampling. Only worth turning on alongside `lam_sw`;
    see grace.models.adapter."""
    severity_film: bool = True


@dataclass
class SamplingConfig:
    k_train: int = 2
    k_eval: int = 8


@dataclass
class LossConfig:
    weighting: str = "jacobian"     # "none" reproduces the GRACE v1 objective
    eps_iso: float = 0.05           # 1.0 is exactly plain MSE
    w_cos: float = 1.0
    w_err: float = 1.0
    lam_sw: float = 0.1
    n_proj: int = 64
    lam_id: float = 0.5
    lam_kl: float = 0.1             # demoted: subsumed by the Jacobian weighting
    kl_temperature: float = 2.0
    lam_sev: float = 0.1


@dataclass
class DiscrepancyConfig:
    hidden: int = 256
    proj: int = 64
    use_severity: bool = True


@dataclass
class TrainConfig:
    """One stage-1 run. `run_id` names its checkpoints and its log."""

    run_id: str
    cache_dir: str
    target_view: str = "clean"      # "clean" (arm B) | "degraded" (arm A control)
    source: str = "cache"           # "cache" | "live"
    epochs: int = 12
    batch_size: int = 256           # features, not images -- go large
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    out_dir: str = "checkpoints/grace/"
    checkpoint_every: int = 0
    """>0 writes intermediate stage-1 checkpoints. Experiment E4 trains the
    discrepancy head against each of them to see whether the adapter erases
    forensic evidence as it improves."""
    detector: str = ""
    """The harness detector config. Needed even in `source: cache` mode: the
    frozen head is what `head_kl` and the Jacobian weighting differentiate
    through, so the model is loaded whether or not the trunk ever runs."""
    split: str = ""
    dataset: str = ""
    log_every: int = 50
    """Diagnostics cadence, in steps: `cos(Δ, j)`, the mean gate and the loss
    terms are computed here and appended to `summary.json`'s history. The W&B
    tracker mirrors the same points, so a dashboard and a summary file never
    disagree about what was measured when."""

    def __post_init__(self):
        # `step % 0` is a ZeroDivisionError several minutes into a run rather
        # than at second zero, and 0 is the natural typo for "never log".
        if self.log_every < 1:
            raise ValueError(
                f"log_every must be >= 1, got {self.log_every}. There is no "
                f"'never': the history in summary.json is written from these "
                f"points. Raise it instead."
            )
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class DiscrepancyTrainConfig:
    """Stage 2. The adapter is frozen; only the aux head and β train.

    Separate from `TrainConfig` because the separation is the scientific claim:
    the label-free adapter is a finished artifact before this stage begins, and
    GRACE and GRACE-D ship the same weights.
    """

    run_id: str
    cache_dir: str
    adapter_checkpoint: str
    epochs: int = 4
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    out_dir: str = "checkpoints/grace/"
    detector: str = ""
    """Needed to score the main logit that the auxiliary logit is fused with."""
    split: str = ""
    dataset: str = ""
    log_every: int = 50
    discrepancy: DiscrepancyConfig = field(default_factory=DiscrepancyConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")


def read_yaml(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build(cls, raw: dict):
    """Instantiate `cls` from `raw`, recursing into nested dataclass fields.

    Unknown keys raise rather than being ignored: a typo in a config that
    silently trains the default objective is a whole day lost.
    """
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"{cls.__name__} got unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        f = known[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _build(f.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_probe_config(path: str | Path) -> ProbeConfig:
    return _build(ProbeConfig, read_yaml(path))


def load_cache_config(path: str | Path) -> CacheConfig:
    return _build(CacheConfig, read_yaml(path))


def load_train_config(path: str | Path) -> TrainConfig:
    return _build(TrainConfig, read_yaml(path))


def load_discrepancy_config(path: str | Path) -> DiscrepancyTrainConfig:
    return _build(DiscrepancyTrainConfig, read_yaml(path))
