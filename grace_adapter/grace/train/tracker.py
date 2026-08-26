"""Weights & Biases logging, as a null object rather than as a conditional.

Every training entry point in this package logs through a `Tracker`. When
tracking is off -- the default -- it is `NullTracker`, whose methods do nothing,
so there is no `if self.wandb is not None` scattered through `loop.py` and no
code path that only executes on someone's machine.

Three rules the rest of the package depends on:

**W&B is never the record.** `summary.json` next to the checkpoints stays the
source of truth for every number, written whether or not anything was tracked. A
run whose dashboard was lost is still fully reportable, and two people comparing
results are comparing files rather than screenshots.

**W&B never fails a run.** A dead network, an expired key or an un-installed
package cannot take down a training job that was going to succeed -- the failure
is warned about once and the run continues untracked. The exception is
`enabled: true` with the package missing, which is a configuration error the
caller must fix rather than a transient one, and is raised at construction
before any GPU time is spent.

**The step axis is the training step**, passed explicitly on every call. W&B's
implicit counter increments per `log()`, which silently interleaves stage 1's
50-step diagnostics with its end-of-epoch rows and makes two runs with different
`log_every` incomparable.

    tracker = build_tracker(cfg.wandb, run_id=..., job_type="stage1", config=...)
    tracker.log({"loss": ...}, step=step)
    tracker.summary({"auc_fused": ...})
    tracker.finish()

Nothing here is imported unless a config asks for it: `import wandb` happens
inside `WandbTracker.__init__`, so `pip install grace-adapter` does not acquire a
tracking dependency.
"""

import warnings
from dataclasses import asdict, is_dataclass


class NullTracker:
    """The default. Every method is a no-op that returns nothing."""

    enabled = False
    url = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        pass

    def summary(self, metrics: dict) -> None:
        pass

    def finish(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return False


class WandbTracker(NullTracker):
    """A live W&B run. Constructed only from `build_tracker`.

    Every logging call is wrapped: a mid-run network failure warns once and
    downgrades the tracker to a no-op for the rest of the job rather than
    raising into the training loop.
    """

    enabled = True

    def __init__(self, cfg, run_id: str, job_type: str, config: dict):
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=cfg.project,
            entity=cfg.entity or None,
            name=run_id,
            group=cfg.group or None,
            job_type=job_type,
            tags=list(cfg.tags),
            mode=cfg.mode,
            config=config,
            reinit=True,        # stage 1 -> stage 2 in one process, and the E4
                                # sweep, both open several runs per interpreter
        )
        self.url = getattr(self._run, "url", None)
        self._dead = False

    def _guard(self, what, *args, **kwargs):
        if self._dead:
            return
        try:
            what(*args, **kwargs)
        except Exception as e:                              # noqa: BLE001
            self._dead = True
            warnings.warn(
                f"W&B logging failed ({e!r}); continuing untracked. The run's "
                f"summary.json is unaffected and remains the record.",
                stacklevel=3,
            )

    def log(self, metrics: dict, step: int | None = None) -> None:
        self._guard(self._run.log, _numeric(metrics), step=step)

    def summary(self, metrics: dict) -> None:
        """Flattened into `run.summary`, which is what the runs table sorts on.

        Nested dicts -- stage 1's per-validation-epoch block, stage 2's per-epoch
        AUCs -- become `validation/epoch_10000/cosine_to_clean` rather than an
        unsortable blob.
        """
        def _set():
            for key, value in _flatten(metrics).items():
                self._run.summary[key] = value

        self._guard(_set)

    def finish(self) -> None:
        self._guard(self._run.finish)
        self._dead = True


def build_tracker(cfg, run_id: str, job_type: str, config=None) -> NullTracker:
    """`WandbTracker` if the config enables it, `NullTracker` otherwise.

    `cfg` may be `None` (a caller that predates the field, or a hand-built
    config in a test), which is the same as disabled.
    """
    if cfg is None or not getattr(cfg, "enabled", False):
        return NullTracker()
    try:
        import wandb  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "wandb.enabled is true but the package is not installed. "
            "`pip install wandb`, or set `wandb: {enabled: false}` in the run "
            "config. This is raised here rather than warned about because the "
            "config asked for tracking explicitly."
        ) from e
    try:
        return WandbTracker(cfg, run_id, job_type, flatten_config(config or {}))
    except Exception as e:                                  # noqa: BLE001
        # Auth and network failures at init are transient; the training run is
        # still worth doing.
        warnings.warn(
            f"could not start a W&B run ({e!r}); continuing untracked. "
            f"`wandb login`, or set mode: offline.",
            stacklevel=2,
        )
        return NullTracker()


def flatten_config(config) -> dict:
    """A run config -- dataclass or dict -- as flat `a/b: value` pairs.

    Flat rather than nested so the W&B runs table can group and filter on
    `loss/lam_sw` directly, which is the whole reason to send the config at all.
    """
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    return _flatten(config)


def _flatten(d, prefix: str = "") -> dict:
    out = {}
    for key, value in (d or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}/"))
        else:
            out[name] = value
    return out


def _numeric(metrics: dict) -> dict:
    """Drop the bookkeeping keys that are axes, not measurements.

    `step` and `epoch` travel inside the same `terms` dict the disk history
    uses; logging `step` as a metric plots a diagonal line and, worse, shadows
    W&B's own step axis.
    """
    return {k: v for k, v in metrics.items() if k not in ("step",)}


# ------------------------------------------------------------------- CLI ----
# Three scripts take the same four flags. Defined here, next to the config they
# write into, so adding a fifth is one edit rather than three that drift.

def add_wandb_args(parser) -> None:
    """Attach `--wandb`, `--wandb-project`, `--wandb-group`, `--wandb-offline`.

    All default to `None`, never to `False`: `apply_wandb_args` only overwrites
    what was passed, so a config that already sets `wandb.enabled: true` is not
    silently switched off by the flag's absence.
    """
    group = parser.add_argument_group("weights & biases")
    group.add_argument(
        "--wandb", dest="wandb", action="store_true", default=None,
        help="log this run to W&B (overrides the config's wandb.enabled)",
    )
    group.add_argument(
        "--no-wandb", dest="wandb", action="store_false", default=None,
        help="disable tracking even if the config enables it",
    )
    group.add_argument("--wandb-project")
    group.add_argument(
        "--wandb-group",
        help="experiment id, e.g. e4_erasure -- what a sweep is compared within",
    )
    group.add_argument(
        "--wandb-offline", action="store_true",
        help="write to ./wandb/ and sync later; for nodes with no outbound network",
    )


def apply_wandb_args(cfg, args) -> None:
    """Fold the parsed flags into `cfg.wandb`, in place.

    Naming a project or a group implies tracking: `--wandb-group e4_erasure`
    without `--wandb` is a request nobody means as "and log nothing".
    """
    wandb_cfg = cfg.wandb
    if getattr(args, "wandb_project", None):
        wandb_cfg.project = args.wandb_project
        wandb_cfg.enabled = True
    if getattr(args, "wandb_group", None):
        wandb_cfg.group = args.wandb_group
        wandb_cfg.enabled = True
    if getattr(args, "wandb_offline", False):
        wandb_cfg.mode = "offline"
        wandb_cfg.enabled = True
    # Explicit --wandb / --no-wandb wins over the implications above.
    if getattr(args, "wandb", None) is not None:
        wandb_cfg.enabled = args.wandb
