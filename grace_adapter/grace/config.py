"""YAML <-> dataclass config, mirroring `load_data.config` / `eval.config`.

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
from grace.splits.base import FeatureSpec
from preprocessing.degrade.crop import POLICIES, SampleCrop
from pipeline.freq.view import FreqExtract


@dataclass
class CropConfig:
    """The multi-scale window shown to the model, in place of a whole image.

    Applied after the degradation recipe and before preprocessing, drawn
    deterministically from `(index, epoch, seed)` so a rendered cache stays
    reproducible. Shared by stage 0, the cache render and live training, because
    a head fit on one window protocol and features rendered under another are not
    the same feature space -- which is what `input_mode` records and
    `_assert_head_matches` enforces at evaluation time.

    Off by default. Every config predating this keeps whole-image behaviour, and
    turning it on is a single key, which is the repo's unit of ablation.

    `s_max` has no default on purpose. The safe upper bound is a property of the
    corpus, not of the method: a window larger than a source cannot be taken, so
    if one class's images are systematically smaller its crops are systematically
    smaller too, and the realized crop size becomes the label. On `wildfake_test`
    a 128-512 range scores E-cropsize 0.9895 -- almost exactly the shortcut the
    crop was introduced to remove. Run `eval_pipeline/scripts/audit_sizes.py`
    against the corpus being trained on and write down what it says.
    """

    enabled: bool = False
    s_min: int = 128
    s_max: int | None = None
    policy: str = "uniform"
    seed: int = 0

    def __post_init__(self):
        if not self.enabled:
            return
        if self.s_max is None:
            raise ValueError(
                "crop.enabled is true but crop.s_max is unset. The safe range is a "
                "property of the corpus, so it has no default: run "
                "eval_pipeline/scripts/audit_sizes.py --config <the training "
                "dataset> and use the s_max it recommends."
            )
        if self.policy not in POLICIES:
            raise ValueError(f"crop.policy must be one of {POLICIES}, got {self.policy!r}")
        if self.s_min < 1 or self.s_max < self.s_min:
            raise ValueError(
                f"need 1 <= crop.s_min <= crop.s_max, got s_min={self.s_min}, "
                f"s_max={self.s_max}"
            )

    def build(self) -> SampleCrop | None:
        """The `(image, index) -> image` the render and training paths apply."""
        if not self.enabled:
            return None
        return SampleCrop(self.s_min, self.s_max, self.seed, self.policy)

    def fingerprint(self) -> str:
        """`CacheSpec.crop_sha`. Empty means whole images, which is a protocol
        rather than a missing value -- see the field's own note."""
        crop = self.build()
        return crop.fingerprint() if crop is not None else ""


@dataclass
class FreqConfig:
    """The patch-DCT side-view rendered alongside the features.

    Off by default, so every cache config predating the frequency branch renders
    exactly what it always did and every cache already on disk stays readable.

    These five numbers are a RENDER-TIME COMMITMENT, and the asymmetry with the
    view count is the reason they are fingerprinted. Views are resumable --
    `build_cache` skips finished ones, so epochs 6-14 can be added to a rendered
    cache later at zero rework. The coefficient set is not: change `patch`,
    `grid` or `radial` and every frequency byte in the directory has to be
    written again. So take the cheap side of the reversible knob (render few
    views) and the safe side of the irreversible one (render full coefficients),
    and let `freq_sha` refuse the mismatch rather than discovering it in a
    training curve.

    `channels` is 3 and is not a knob: `extract_freq` reads whatever the decoded
    RGB image has, and the field exists so the declared shape and the fingerprint
    can be computed without decoding one. Chromatic artefacts are among the
    things a generation trace shows up in, so collapsing to luma would be a
    deliberate ablation rather than a saving -- and it is not one this ships.
    """

    enabled: bool = False
    patch: int = 8
    """DCT block side. 8 is JPEG's, so the block-boundary artefacts the JPEG
    degradation family produces land on coefficients this basis resolves."""
    grid: int = 14
    """Cells per side after adaptive pooling. 14x14 = 196, one per DINOv3 patch
    token at 224, and fixed across the whole 128-512px crop range -- which is
    what makes one rendered view serve every scale the draw produces."""
    channels: int = 3
    radial: bool = True
    """Order coefficients by radial spatial frequency, so a band is a contiguous
    slice of the coefficient axis. The enricher's band masks and E13's top-k
    both assume it."""
    norm: str = "log1p"

    def __post_init__(self):
        if not self.enabled:
            return
        if self.patch < 2 or self.patch & (self.patch - 1):
            raise ValueError(
                f"freq.patch must be a power of two >= 2, got {self.patch}. The "
                f"DCT is taken per block and 8 is JPEG's block size."
            )
        if self.grid < 1:
            raise ValueError(f"freq.grid must be >= 1, got {self.grid}")
        if self.channels not in (1, 3):
            raise ValueError(f"freq.channels must be 1 or 3, got {self.channels}")
        if self.norm != "log1p":
            raise ValueError(
                f"freq.norm: only 'log1p' is implemented, got {self.norm!r}. It is "
                f"a field rather than a constant because it is fingerprinted -- "
                f"adding a second one must invalidate every rendered view."
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.grid * self.grid, self.channels * self.patch * self.patch)

    def feature(self) -> FeatureSpec | None:
        """`CacheSpec.freq_feature`. `tokens` layout: the cell axis is a group
        axis exactly as a `tokens` seam's patch axis is, which is what lets
        `ShardWriter` and `_gather` carry it with no new on-disk format."""
        if not self.enabled:
            return None
        return FeatureSpec(layout="tokens", shape=self.shape)

    def build(self) -> FreqExtract | None:
        """The `(image) -> tensor` the render path applies in its workers."""
        if not self.enabled:
            return None
        return FreqExtract(patch=self.patch, grid=self.grid, radial=self.radial)

    def fingerprint(self) -> str:
        """`CacheSpec.freq_sha`. Empty when disabled, which pairs with a
        `freq_feature` of None -- the two are set and cleared together."""
        extract = self.build()
        return "" if extract is None else extract.fingerprint(self.channels, self.norm)


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

    grid_file: str = "preprocessing/configs/degradations.yaml"
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

    `group` is what makes the sweeps in README section 6 legible: set it to the
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
    val_dataset: str | list[str] = ""
    """Held-out images for model selection. One config path, or several.

    A list because selection has taken more than one dataset before: under NTIRE
    it ran against val + val_hard, and one scalar had to come out of both, so the
    selection score is the unweighted mean of their AUCs and a head that won by
    collapsing on the hard set could not be selected. Per-dataset AUC and
    accuracy are reported alongside it either way.

    Today it holds ONE entry -- wildfake_train_val, the held-out split of the
    training manifest -- so the mean is over a single set and that protection is
    inactive. Whatever it holds, it must never name the reported benchmark.
    """
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
    crop: CropConfig = field(default_factory=CropConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def val_datasets(self) -> list[str]:
        """`val_dataset` as a list, however it was written."""
        if not self.val_dataset:
            return []
        if isinstance(self.val_dataset, str):
            return [self.val_dataset]
        return list(self.val_dataset)


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
    batch_size: int = 8
    """IMAGES per batch, and one image is now every view of it.

    A worker decodes once and returns `1 + n_epochs + n_val_epochs` preprocessed
    views, so in-flight memory is `batch_size x n_views x input`, not
    `batch_size x input`. Small on purpose: 8 images at 15 views of 3x224x224 is
    72 MB a batch before prefetching. `trunk_batch_size`, not this, is what
    keeps the GPU efficient."""
    trunk_batch_size: int = 128
    """Samples per trunk forward. The views of a batch are flattened and fed in
    chunks of this, so a small `batch_size` does not starve the GPU: DINOv3
    ViT-S/16 measures 715 img/s at 32 and 1006 at 128."""
    num_workers: int = 8
    """Decode, degrade and preprocess all happen here. This is the knob that
    sets render throughput -- see `grace.cache.writer`."""
    max_images: int | None = None
    device: str = "auto"
    split_args: dict = field(default_factory=dict)
    """Keyword arguments for the split, e.g. `{tap_blocks: [0, 2, 4, 6, 9]}`.

    The split is named by dotted path and built with no arguments today, so
    anything that varies *per run* rather than per detector has to arrive here.
    The ladder's tap set is the first such thing: two runs over one detector can
    tap different blocks, and the cache each renders is only valid for its own.
    `CacheSpec.taps` records what was actually rendered, so a train run against a
    mismatched cache is refused rather than silently mis-fed.
    """
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    freq: FreqConfig = field(default_factory=FreqConfig)
    """Render the DCT side-view too. See `FreqConfig`.

    Note what this does NOT change: the number of views. A frequency cache is a
    separate render with its own `out_dir` and its own (smaller) `n_epochs`,
    because the frequency branch trains a small module over a few epochs while
    stage 1 wants fifteen -- and because the coefficient set is the irreversible
    knob, so the view count is where the saving is taken.
    """


@dataclass
class AdapterConfig:
    bottleneck: int = 256
    n_blocks: int = 2
    per_channel_gate: bool = True
    dropout: float = 0.0
    severity_film: bool = True
    taps: bool = False
    """Build a `LadderAdapter` reading the split's intermediate taps.

    A bool, not the block list: *which* blocks are tapped is a property of the
    split (`split_args.tap_blocks`) and of the cache rendered from it, and
    stating it twice is an invitation for the two to disagree. This says only
    whether the adapter reads them. `grace.models.ladder.tap_spec_for` is where
    the intent and the split's actual taps are reconciled.
    """
    tap_dim: int = 64
    """Width each tap is projected to. Drives the ladder's parameter count
    almost entirely -- see the budget table in `grace.models.ladder`."""
    gate_init: float = -4.0
    """Logit the seam gate -- and the ladder's `tap_gate` -- starts at.

    A LOGIT, not a gate value: sigmoid(-4) ~= 0.018. Small so the adapter is a
    near-no-op at step 0 even if the exact-identity guarantee ever breaks, and
    non-zero so gradient reaches the gate (and, in the ladder, `tap_proj` behind
    it) from the first step. Sweeping it is an experiment about the second of
    those -- see `grace.models.adapter.GATE_INIT`."""


@dataclass
class LossConfig:
    weighting: str = "jacobian"     # "none" makes L_err plain F.mse_loss. It does
                                    # NOT reproduce the GRACE v1 objective: v1 also
                                    # ran lam_kl 0.5 and the since-removed lam_sw /
                                    # n_proj / lam_id terms. See EXPERIMENTS.md §8.
    eps_iso: float = 0.05           # 1.0 is exactly plain MSE
    w_cos: float = 1.0
    w_err: float = 1.0
    lam_kl: float = 0.1             # demoted: subsumed by the Jacobian weighting
    kl_temperature: float = 2.0
    lam_sev: float = 0.1


@dataclass
class DiscrepancyConfig:
    hidden: int = 256
    proj: int = 64
    use_severity: bool = True
    use_taps: bool = False
    """Feed the ladder's per-tap drift norms to the head, alongside Δ.

    Requires a ladder stage-1 checkpoint -- it is the ladder that computes the
    per-tap read, and stage 2 never rebuilds the adapter. False against a ladder
    is the control this arm is read against; True against a plain adapter is a
    config error, raised at startup rather than silently ignored.

    The point is the `vector` seam. On DINOv3 the head sees ONE drift norm no
    matter how deep the damage entered, which is the weakest form of the RA-Det
    argument; the tap norms are the per-block damage profile a `layers` seam
    would have supplied for free. See `grace.models.discrepancy`."""
    lam_aux: float = 1.0
    """Weight on the auxiliary head's OWN supervised loss. 0 restores the
    fused-only objective.

    Without it the aux head is trained solely through `beta * aux`, which has
    two consequences. `beta` initializes to 0, so the aux head receives exactly
    zero gradient on step one and the branch bootstraps off `beta` reacting to
    aux's random initialization -- making the learned sign a coin flip per run
    (E4 produced 4 negative and 2 positive). And `loss` depends on the pair only
    through the product, so `(beta, aux)` and `(-beta, -aux)` are exactly
    equivalent: nothing breaks the tie.

    Supervising the aux head directly pins the sign to the labels, gives it
    gradient regardless of `beta`, and -- the reason it matters for E4 -- makes
    `auc_aux` measure the question the experiment asks. Trained fused-only, the
    aux head learns whatever residual helps a main head that is already at
    ~0.9999 AUC on held-out degradations, which is nearly no signal at all. E4
    asks how much forensic evidence the drift carries *standalone*, and only a
    standalone objective measures that.

    `beta` is left unconstrained: the sign degeneracy is broken by supervision
    here, not by reparameterizing the fusion.
    """


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
    decay_gate: bool = True
    """Apply `weight_decay` to the gate logits along with everything else.

    True is what every run before this was trained under. False exempts
    `gate_logit`/`tap_gate_logit` and nothing else: decoupled decay pulls a logit
    toward 0, i.e. the gate toward 0.5, so with it on the gate opens whether or
    not the objective asks it to -- see `grace.train.loop._param_groups`."""
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
    val_every: int = 0
    """>0 runs the held-out validation block every N epochs during training,
    rather than only once at the end.

    Same `epoch % N == 0` convention as `checkpoint_every`, over the same cache
    epoch ids, so setting both to the same value pairs every intermediate
    checkpoint with the numbers it scored -- which is what an E4 curve wants.
    The final validation is unconditional either way, so `val_every: 0` (the
    default) is exactly the old behaviour.

    Not free: each pass scores every held-out degradation epoch and every row of
    every `val_datasets` cache once through. Seconds on the PoC
    cache; worth timing against the epoch before setting it to 1 on a large one.
    """
    detector: str = ""
    """The harness detector config. Needed even in `source: cache` mode: the
    frozen head is what `head_kl` and the Jacobian weighting differentiate
    through, so the model is loaded whether or not the trunk ever runs."""
    split: str = ""
    split_args: dict = field(default_factory=dict)
    """Keyword arguments for the split, e.g. `{tap_blocks: [0, 2, 4, 6, 9]}`.

    The split is named by dotted path and built with no arguments today, so
    anything that varies *per run* rather than per detector has to arrive here.
    The ladder's tap set is the first such thing: two runs over one detector can
    tap different blocks, and the cache each renders is only valid for its own.
    `CacheSpec.taps` records what was actually rendered, so a train run against a
    mismatched cache is refused rather than silently mis-fed.
    """
    dataset: str = ""
    val_datasets: list[str] = field(default_factory=list)
    """Held-out IMAGES for stage-1 validation. Parallel to `val_cache_dirs`.

    Distinct from the cache's own validation axis, which holds out
    *degradations* (`val_epochs`, ids from 10000) over the training images. That
    axis answers "does this adapter generalize to an unseen corruption"; it
    cannot answer "does it generalize to an unseen image", because every row it
    scores was trained on. These datasets answer the second question, and both
    are reported separately in summary.json under `validation`.
    """
    val_cache_dirs: list[str] = field(default_factory=list)
    """Rendered cache root per entry of `val_datasets`, same order.

    Separate roots because `build_cache.py` derives its root from the DETECTOR
    name alone, so two datasets rendered for one detector would collide -- and
    the second render would be rejected on `manifest_sha` rather than silently
    mixing. One `out_dir` per dataset keeps them apart.
    """
    log_every: int = 50
    """Diagnostics cadence, in steps: `cos(Δ, j)`, the mean gate and the loss
    terms are computed here and appended to `summary.json`'s history. The W&B
    tracker mirrors the same points, so a dashboard and a summary file never
    disagree about what was measured when."""

    def __post_init__(self):
        if len(self.val_datasets) != len(self.val_cache_dirs):
            raise ValueError(
                f"val_datasets has {len(self.val_datasets)} entry(s) but "
                f"val_cache_dirs has {len(self.val_cache_dirs)}. They are parallel "
                f"lists -- each dataset needs the cache root it was rendered to."
            )
        # Negative would silently never fire -- `% 0` itself is guarded by the
        # `and` at the call site, which is what makes 0 the "end only" value.
        if self.val_every < 0:
            raise ValueError(
                f"val_every must be >= 0, got {self.val_every}. 0 means "
                f"'validate once, at the end of the run'."
            )
        # `step % 0` is a ZeroDivisionError several minutes into a run rather
        # than at second zero, and 0 is the natural typo for "never log".
        if self.log_every < 1:
            raise ValueError(
                f"log_every must be >= 1, got {self.log_every}. There is no "
                f"'never': the history in summary.json is written from these "
                f"points. Raise it instead."
            )
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    crop: CropConfig = field(default_factory=CropConfig)
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
    split_args: dict = field(default_factory=dict)
    """Keyword arguments for the split, e.g. `{tap_blocks: [0, 2, 4, 6, 9]}`.

    The split is named by dotted path and built with no arguments today, so
    anything that varies *per run* rather than per detector has to arrive here.
    The ladder's tap set is the first such thing: two runs over one detector can
    tap different blocks, and the cache each renders is only valid for its own.
    `CacheSpec.taps` records what was actually rendered, so a train run against a
    mismatched cache is refused rather than silently mis-fed.
    """
    dataset: str = ""
    val_datasets: list[str] = field(default_factory=list)
    """Held-out IMAGE sets, same field as TrainConfig's and scored the same way.

    E4 is unreadable without them: on `held_out_degradations` the base head is
    already at ~0.9999 AUC, so `auc_fused` cannot beat `auc_main` by more than
    rounding regardless of what the discrepancy branch learned. These sets are
    where the main head has headroom.
    """
    val_cache_dirs: list[str] = field(default_factory=list)
    log_every: int = 50
    discrepancy: DiscrepancyConfig = field(default_factory=DiscrepancyConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    """Stage 2 never decodes an image -- it reads cached features throughout --
    but it still has to name the window protocol, because `crop_sha` is what
    stops it reading a whole-image cache as though the stage-1 adapter it is
    freezing had been trained on one. Set it to whatever stage 1 used."""
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")


@dataclass
class EnricherConfig:
    """The frequency enricher's geometry. Its OWN config, not `AdapterConfig`'s.

    Deliberately a separate dataclass with a separate checkpoint, following
    `DiscrepancyConfig`. `load_adapter` rebuilds a stage-1 module with
    `AdapterConfig(**payload["adapter_cfg"])`, so every field on that dataclass
    is a checkpoint-compatibility surface: adding one there would brick all 25
    existing adapter checkpoints the same way `noise_dim` did. Adding one here
    costs nothing, because nothing has been trained against it yet.
    """

    d_model: int = 256
    """Width of the attention. The queries are projected down from the 768-d
    seam, so this is the enricher's capacity knob and almost all of its
    parameters."""
    n_heads: int = 4
    n_bands: int = 2
    """Band experts. 2 = the HF/LF pair, 1 = the single-branch control (E11).

    Two because blur-type damage (high frequencies destroyed) and noise-type
    damage (high frequencies ADDED) move the same coefficients in opposite
    directions, and one expert reading both has to represent that with a single
    set of weights and a single gate. Each expert owns its band mask, its K/V
    projection, its output projection and its gate, so the two can specialise
    completely -- including to the point of one of them staying shut, which is
    what makes E11 readable.
    """
    dropout: float = 0.0
    gate_init: float = -4.0
    """Logit each expert's gate starts at; sigmoid(-4) ~= 0.018. Belt to the
    zero-initialised output projections' braces -- the exact identity comes from
    the projections, not from this (see `grace.models.frequency`)."""
    severity_film: bool = True
    """Condition the gates on the predicted severity, reusing stage 1's scalar.
    'How much should I trust the frequency read' is a different question at
    JPEG-90 and JPEG-30, and severity is the only input that distinguishes
    them."""
    pos_emb: bool = True
    """Learned per-cell position embedding. Off is one half of E13: without it
    the 196 cells are an unordered set and the attention cannot express 'the
    high-band energy in the top-left corner', only 'somewhere'."""
    top_k: int | None = None
    """Keep only the `k` lowest radial frequencies PER CHANNEL. None = all.

    The other half of E13. Meaningful only because `radial: true` makes a
    frequency band a contiguous slice, so 'top-k' means the k lowest
    frequencies rather than the k first in raster order. Applied at read time
    over the full rendered view, so sweeping it costs no re-render -- which is
    exactly why the cache stores full coefficients.
    """
    learn_masks: bool = True
    """Let the band masks move off their hard initialisation. False pins them
    to `band_masks`, making the split a fixed decomposition rather than a
    learned one."""

    def __post_init__(self):
        if self.n_bands < 1:
            raise ValueError(f"enricher.n_bands must be >= 1, got {self.n_bands}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"enricher.d_model ({self.d_model}) must divide evenly by n_heads "
                f"({self.n_heads})"
            )
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"enricher.top_k must be >= 1 or null, got {self.top_k}")


@dataclass
class EnrichTrainConfig:
    """Stage 2 for the frequency branch. The adapter is frozen, as in GRACE-D.

    Separate from `DiscrepancyTrainConfig` because they train different modules
    against different inputs, and sharing one dataclass would put every field of
    each on the other. What they DO share is the invariant: the label-free
    adapter is a finished artifact before either begins, and both ship it
    unchanged.
    """

    run_id: str
    cache_dir: str
    adapter_checkpoint: str
    epochs: int = 4
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    lam_anchor: float = 0.1
    """Weight on `||fused - f_corrected||`. 0 is E15's unanchored arm.

    The enricher is trained supervised against labels, so nothing in the
    objective otherwise stops it from walking the features somewhere the frozen
    head happens to separate on this corpus -- which is a content classifier
    wearing the adapter's output as a starting point. The anchor says the
    enrichment is a correction to the corrected features, not a replacement for
    them.
    """
    finetune_head: bool = False
    """Train a COPY of the detector's head alongside the enricher. E14's second
    arm.

    Not a default, and shipped as a first-class arm rather than an afterthought,
    because round 1 measured `parallel_fraction = 0.0298`: 97% of feature drift
    lay in directions the frozen head does not read. If that holds here, a frozen
    head reduces the whole cross-attention enricher to a scalar logit shift --
    the same function class as GRACE-D's aux logit, at a thousand times the
    parameter count. Read E0-drift on this corpus BEFORE choosing, and read
    `parallel_fraction` as the predicted ceiling on the frozen arm.

    What the frozen arm buys is the head's provenance: it is the head the
    baseline was measured with, so a gain is attributable to the features. The
    fine-tuned arm gives that up. Report both.
    """
    head_lr: float = 1e-4
    """Learning rate for the head copy when `finetune_head`. An order below the
    enricher's: the head arrives trained, and the arm is asking whether it can be
    *adjusted* to read enriched features, not re-fit from scratch."""
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    out_dir: str = "checkpoints/grace/"
    detector: str = ""
    """The harness detector config. Needed even though no trunk runs: the frozen
    head is what the BCE is taken through."""
    split: str = ""
    split_args: dict = field(default_factory=dict)
    dataset: str = ""
    val_datasets: list[str] = field(default_factory=list)
    val_cache_dirs: list[str] = field(default_factory=list)
    log_every: int = 50
    enricher: EnricherConfig = field(default_factory=EnricherConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    """Stage 2 decodes no images, but it still names the window protocol: it is
    `crop_sha` that stops it reading a whole-image cache as though the stage-1
    adapter it is freezing had been trained on one. Set it to whatever stage 1
    used."""
    freq: FreqConfig = field(default_factory=FreqConfig)
    """The extraction protocol this run expects the cache to carry, and the one
    stamped into the enricher checkpoint. Checked against `spec.freq_sha` at
    startup: same shape, different frequencies is the mismatch that would
    otherwise train quietly and score nonsense, because the band masks are
    indexed by position along the coefficient axis."""
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if len(self.val_datasets) != len(self.val_cache_dirs):
            raise ValueError(
                f"val_datasets has {len(self.val_datasets)} entry(s) but "
                f"val_cache_dirs has {len(self.val_cache_dirs)}. They are parallel "
                f"lists -- each dataset needs the cache root it was rendered to."
            )
        if not self.freq.enabled:
            raise ValueError(
                "freq.enabled must be true for an enrichment run -- the whole "
                "stage reads the frequency view. Point cache_dir at a cache "
                "rendered with configs/cache/wildfake_freq.yaml and set the same "
                "`freq:` block here."
            )
        if self.lam_anchor < 0:
            raise ValueError(f"lam_anchor must be >= 0, got {self.lam_anchor}")


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


def load_enrich_config(path: str | Path) -> EnrichTrainConfig:
    return _build(EnrichTrainConfig, read_yaml(path))
