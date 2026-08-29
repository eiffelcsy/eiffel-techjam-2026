"""The two training stages.

**Stage 1 -- the adapter.** Label-free. In `source: cache` mode a step is:

    f_deg, f_clean = batch                     # two memmap reads
    f_adapted      = adapter(f_deg.float())    # ~2M params
    loss           = total_loss(...)           # + one frozen head for head_kl
    loss.backward(); opt.step(); ema.update()

No image decode, no augmentation, no trunk. A full run costs minutes, which is
the point of pre-rendering: the seed sweep, the geometry grid and the weight soup
that normally get cut for time become affordable, and any ablation can be re-run
whenever a question about it arises.

**Stage 2 -- the discrepancy head.** Supervised, adapter frozen. Seconds, because
Δ is deterministic given cached features and a frozen adapter. That cheapness is
what makes experiment E4 possible: run stage 2 against every stage-1 checkpoint
and watch whether the auxiliary head's standalone AUC *falls* as the adapter
improves. If it does, the adapter is provably erasing forensic evidence, and the
retention-versus-drift-preservation curve is the figure.

Invariants asserted every step, not once at startup:

  * the split is in eval mode with no trainable parameters (`assert_frozen`)
  * cached features are cast to float32 before any loss -- fp16 MSE on
    unnormalized ViT features underflows to zero and trains nothing
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from grace.cache.reader import FeatureCache
from grace.cache.spec import CacheSpec
from grace.cache.schedule import val_epochs
from grace.models.discrepancy import FusedHead
from grace.models.factory import (
    build_adapter, build_discrepancy_head, build_severity_head, load_adapter, save_adapter,
)
from grace.models.ladder import tap_spec_for
from grace.train import diagnostics as D
from grace.train.data import build_loader
from grace.train.ema import EMA
from grace.train.losses import supervised_bce, total_loss
from grace.train.tracker import build_tracker, flatten, flatten_config
from grace.train.weighting import head_gradient
from pipeline.config import load_dataset_config
from pipeline.data.manifest import load_manifest
from pipeline.eval.metrics import (
    error_breakdown, retention, roc_auc, threshold_from_clean,
)
from pipeline.utils.seeding import seed_everything


def cosine_with_warmup(opt, warmup: int, total: int) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(opt, fn)


def _to_float(t: torch.Tensor, device) -> torch.Tensor:
    """Cast out of the cache dtype. See the module docstring."""
    return t.to(device=device, dtype=torch.float32)


def _cache_loader_cfg(cfg):
    """A loader spec pinned to `source: cache`.

    Validation and stage 2 always read pre-rendered features, even when the run
    being validated trained in `live` mode -- otherwise the held-out numbers
    would be measured against a different set of degradations than the one
    `val_epochs` reserved.
    """
    return type("_LoaderCfg", (), {
        "source": "cache",
        "batch_size": cfg.batch_size,
        "num_workers": 0,
    })()


def _expect_spec(split, tap_spec, crop_sha: str = "") -> CacheSpec:
    """What this run needs a cache to be, for `CacheSpec.assert_compatible`.

    `tap_spec` is passed in rather than derived: stage 1 gets it from its
    adapter config, stage 2 from the checkpoint it is about to score, and those
    are genuinely different questions about the same cache.

    Only the parts that vary per run go in: the feature layout, the taps, and the
    crop. The four fingerprints are left blank because the *cache* is the
    authority on those -- a run does not know the manifest hash it wants, it
    knows the split it is training against. Naming the taps here is what turns
    "ladder trained on a cache rendered for other blocks" from a silent shape
    coincidence into a refusal at startup.

    `crop_sha` is the exception to "the cache is the authority", and it is passed
    rather than blank for the same reason the taps are named: the run genuinely
    does know which window protocol it wants, because its own `crop:` block
    determines what `source: live` will draw and what the clean targets have to
    be windows of. A blank here would make every cropped cache readable by a
    whole-image run.
    """
    return CacheSpec(
        detector=split.name,
        feature=split.feature_spec,
        n=0,
        crop_sha=crop_sha,
        taps=split.taps() if tap_spec is not None else (),
        tap_feature=tap_spec,
    )


def _load_val_sets(cfg, spec, expect: CacheSpec | None = None) -> list:
    """Held-out IMAGES, each from its own rendered cache root.

    Loaded up front so a missing or mis-specified val cache fails now rather
    than after the full training run, at the one moment its result is wanted.

    Shared by both stages: stage 2 scores the same held-out images as stage 1,
    against the same integrity checks, so an E4 sweep and the stage-1 retention
    curve are read off the same axis rather than two that merely look alike.
    """
    val_sets = []
    for ds_path, cache_dir in zip(
        getattr(cfg, "val_datasets", []) or [], getattr(cfg, "val_cache_dirs", []) or []
    ):
        ds_cfg = load_dataset_config(ds_path)
        # Same tap check as the training cache: a val cache rendered without
        # taps would fail per-batch inside `_alignment`, after training.
        val_cache = FeatureCache(cache_dir, expect=expect)
        val_manifest = load_manifest(ds_cfg.manifest, ds_cfg.split)
        if len(val_manifest) != val_cache.spec.n:
            raise RuntimeError(
                f"{cache_dir} holds {val_cache.spec.n} rows but {ds_path} selects "
                f"{len(val_manifest)}. The cache was rendered from a different "
                f"manifest -- re-render it with scripts/build_cache.py."
            )
        if (val_cache.spec.feature.layout, val_cache.spec.feature.shape) != (
            spec.layout, spec.shape
        ):
            raise RuntimeError(
                f"{cache_dir} holds {val_cache.spec.feature.layout}"
                f"{val_cache.spec.feature.shape} but the detector emits "
                f"{spec.layout}{spec.shape}. Re-render this val cache against "
                f"the detector being trained."
            )
        val_sets.append((ds_cfg.name, val_cache, val_manifest))
    return val_sets


def _param_groups(adapter, severity_head, cfg) -> list[dict]:
    """AdamW groups, splitting the gate logits out when `decay_gate` is off.

    Decoupled weight decay pulls a *logit* toward zero, which for the gate means
    toward 0.5 -- it opens the gate on its own, with no help from the objective.
    That is not a hypothetical: measured on `dinov3_sweep_gate_-4`, decay alone
    accounts for more than the whole of the gate's drift over 12 epochs, and the
    alignment term's net pull is the other way. So "the gate climbed off init",
    the health signal this project reads first, was reporting the optimizer.

    `decay_gate: false` puts `gate_logit` and `tap_gate_logit` in a group with no
    decay and leaves every other parameter where it was, so the arm differs from
    its control in the gate alone. Default is `true`, which reproduces the single
    group the runs before this were trained under, byte for byte.
    """
    params = list(adapter.parameters()) + (
        list(severity_head.parameters()) if severity_head else []
    )
    if cfg.decay_gate:
        return [{"params": params}]

    gates = {
        id(p): p for n, p in adapter.named_parameters() if n.endswith("gate_logit")
    }
    if not gates:
        raise RuntimeError(
            "decay_gate is false but the adapter has no gate_logit to exempt -- "
            "the name this selects on must have changed."
        )
    rest = [p for p in params if id(p) not in gates]
    return [
        {"params": rest},
        {"params": list(gates.values()), "weight_decay": 0.0},
    ]


def train_adapter(cfg, split, manifest, schedule) -> dict:
    """Stage 1. Returns the summary written next to the checkpoints.

    The target-view branch is the entire arm A / arm B ablation:

        cfg.target_view == "clean"     -> f_clean from the cache   (arm B, proposed)
        cfg.target_view == "degraded"  -> f_deg.detach()           (arm A, control)

    Arm A is symmetric self-distillation and should do nothing useful. If it
    matches arm B, the asymmetry -- the clean view as teacher -- was not the
    mechanism, and that is worth knowing on day 2 rather than day 5.
    """
    if cfg.target_view not in ("clean", "degraded"):
        raise ValueError(f"target_view must be 'clean' or 'degraded', got {cfg.target_view!r}")
    seed_everything(cfg.seed)
    device = next(split.parameters()).device
    spec = split.feature_spec

    # The split decides the tap shape; `cfg.adapter.taps` only says whether to
    # read them. Resolved before the cache is opened so a ladder run against a
    # tapless cache is refused at startup, not several thousand steps in.
    tap_spec = tap_spec_for(split, cfg.adapter)
    expect = _expect_spec(split, tap_spec, cfg.crop.fingerprint())
    cache = FeatureCache(cfg.cache_dir, expect=expect)

    val_sets = _load_val_sets(cfg, spec, expect)

    adapter = build_adapter(spec, cfg.adapter, tap_spec, split.taps()).to(device)
    severity_head = (
        build_severity_head(spec).to(device) if cfg.loss.lam_sev > 0 else None
    )
    params = list(adapter.parameters()) + (
        list(severity_head.parameters()) if severity_head else []
    )
    ema = EMA(adapter, cfg.ema_decay)
    opt = torch.optim.AdamW(
        _param_groups(adapter, severity_head, cfg),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    epochs = [e for e in cache.epochs() if e < min(val_epochs(1))][: cfg.epochs]
    if not epochs:
        raise RuntimeError(f"no rendered training epochs under {cfg.cache_dir}")
    steps_per_epoch = len(manifest) // cfg.batch_size
    sched = cosine_with_warmup(opt, cfg.warmup_steps, steps_per_epoch * len(epochs))

    out_dir = Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess = split.preprocess_fn() if cfg.source == "live" else None

    # Off unless a config asks for it, and a no-op object rather than a None to
    # branch on. `arm` is logged as config because target_view IS the E2
    # ablation and grouping a sweep by it is the first thing anyone does.
    tracker = build_tracker(
        cfg.wandb, run_id=cfg.run_id, job_type="stage1",
        config={**flatten_config(cfg), "detector_name": split.name,
                "feature_layout": spec.layout, "feature_dim": spec.dim,
                "adapter_params": sum(p.numel() for p in adapter.parameters()),
                "arm": "B_clean_teacher" if cfg.target_view == "clean" else "A_self"},
    )

    step, history, val_history = 0, [], []
    for epoch in epochs:
        loader = build_loader(
            cfg, cache, manifest, schedule, epoch, preprocess,
            with_taps=tap_spec is not None,
            # Only reaches `source: live`, and there it must be the crop the
            # cache's clean view was rendered under: `f_clean` is the target for
            # whatever window `image` shows. `crop_sha` on the cache is what
            # catches a run that gets this wrong.
            crop=cfg.crop.build(),
        )
        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            split.assert_frozen()

            f_clean = _to_float(batch["f_clean"], device)
            taps_deg = None
            if cfg.source == "live":
                with torch.no_grad():
                    # One forward for both, whether or not taps are wanted --
                    # `trunk_with_taps` returns (f, None) for a tapless split.
                    f_deg, taps_deg = split.trunk_with_taps(batch["image"].to(device))
                    f_deg = f_deg.float()
                    taps_deg = taps_deg.float() if taps_deg is not None else None
            else:
                f_deg = _to_float(batch["f_deg"], device)
                if tap_spec is not None:
                    taps_deg = _to_float(batch["taps_deg"], device)
            target = f_clean if cfg.target_view == "clean" else f_deg.detach()

            sev_target = batch["severity"].to(device).float()
            sev_pred = severity_head(f_deg) if severity_head else None
            # Predicted severity on half the steps: at inference it is predicted,
            # never given, and training only on ground truth teaches the adapter
            # to trust an input it will not have.
            sev_in = sev_target
            if sev_pred is not None and step % 2 == 1:
                sev_in = sev_pred.detach()

            j = head_gradient(split.head, target) if cfg.loss.weighting == "jacobian" else None
            f_adapted = adapter(f_deg, severity=sev_in, taps=taps_deg)

            logging_step = step % cfg.log_every == 0
            loss, terms = total_loss(
                head=split.head,
                f_adapted=f_adapted, f_clean=target, j=j,
                severity_pred=sev_pred, severity_target=sev_target,
                cfg=cfg.loss, diagnostics=logging_step,
            )
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            if logging_step:
                # Read HERE and not in the diagnostics block below: `opt.step()`
                # is followed by `zero_grad`, so by then there is nothing left to
                # read. The total is free -- clipping computes it anyway -- and
                # `tap_proj` is the parameter the non-zero gate init exists to
                # keep fed, so a gate_init sweep is read on these two.
                terms["grad_norm"] = float(total_norm)
                if adapter.reads_taps and adapter.tap_proj.weight.grad is not None:
                    terms["grad_norm/tap_proj"] = float(
                        adapter.tap_proj.weight.grad.norm()
                    )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            ema.update(adapter)

            if logging_step:
                with torch.no_grad():
                    terms["gate"] = float(adapter.gate().mean().detach())
                    if adapter.reads_taps:
                        # "How much does the correction lean on block k", per
                        # step. This is the figure the RINE per-layer gate
                        # promised, on a detector that does not need a `layers`
                        # head to produce it.
                        terms.update(
                            {f"tap_gate/{n}": v
                             for n, v in adapter.tap_weights().items()}
                        )
                    # cos_decision is E3's readout, so it has to exist in the
                    # unweighted arm too -- that arm is the one the figure is
                    # about. Build a Jacobian here when the loss did not need
                    # one; head_gradient runs on its own graph and detaches, so
                    # this stays out of the adapter's and costs one head
                    # backward per logging step rather than per step.
                    j_diag = j if j is not None else head_gradient(split.head, target)
                    terms["cos_decision"] = float(
                        D.decision_alignment(f_adapted, f_deg, j_diag).abs().mean()
                    )
                    terms["step"] = step
                    terms["epoch"] = epoch
                    terms["lr"] = float(sched.get_last_lr()[0])
                    history.append(dict(terms))
                    # Same points as the disk history, so a dashboard and a
                    # summary.json can never disagree about what was measured.
                    tracker.log({f"train/{k}": v for k, v in terms.items()
                                 if k != "step"}, step=step)
            step += 1

        # Intermediate checkpoints exist for experiment E4: stage 2 is trained
        # against each of them, and a falling auxiliary AUC as stage 1 improves
        # is direct evidence that restoring features erases forensic evidence.
        if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
            save_adapter(
                out_dir / f"step_{step:06d}.pt", adapter, spec, cfg.adapter,
                extra={
                    "step": step,
                    **({"severity_state_dict": severity_head.state_dict()}
                       if severity_head is not None else {}),
                },
            )

        # Held-out numbers *during* the run, when a config asks for them. Same
        # `validate` call as the final one -- one measurement, so a mid-run row
        # and the summary's `validation` block are directly comparable -- and
        # the same epoch convention as `checkpoint_every` above, so a matching
        # cadence pairs each checkpoint with the numbers it scored.
        #
        # `epoch=epoch` scores the held-out IMAGES on the draw training just
        # finished, one pass per val cache rather than a sweep over all of them.
        # The held-out DEGRADATION axis is unaffected -- it is the val-numbered
        # epochs of the training cache and is always scored in full.
        if cfg.val_every and epoch % cfg.val_every == 0:
            # Forked RNG. Every DataLoader validate builds draws a base seed
            # from the global generator -- shuffle=False and num_workers=0 do
            # not exempt it -- so validating in-line would shift the training
            # loader's own stream and two runs differing only in `val_every`
            # would stop being seed-for-seed comparable, which is the one
            # property the whole arm A / arm B design rests on.
            with torch.random.fork_rng(
                devices=[device] if device.type == "cuda" else []
            ):
                scores = validate(
                    cfg, adapter, split, cache, manifest, severity_head, val_sets,
                    epoch=epoch,
                )
            val_history.append({"epoch": epoch, "step": step, **scores})
            tracker.log(
                {f"val/{k}": v for k, v in flatten(scores).items()}, step=step
            )

    # The severity head travels with the adapter: at inference severity is
    # predicted, so a FiLM-conditioned adapter shipped without it would silently
    # fall back to the unconditioned gate.
    extra = {"step": step}
    if severity_head is not None:
        extra["severity_state_dict"] = severity_head.state_dict()

    save_adapter(out_dir / "last.pt", adapter, spec, cfg.adapter, extra=extra)
    ema_adapter = build_adapter(spec, cfg.adapter, tap_spec, split.taps())
    ema.copy_to(ema_adapter)
    save_adapter(out_dir / "ema.pt", ema_adapter, spec, cfg.adapter, extra=extra)

    summary = {
        "run_id": cfg.run_id,
        "target_view": cfg.target_view,
        "source": cfg.source,
        "steps": step,
        "epochs": epochs,
        "history": history,
        # Always present and always the finished adapter, whatever `val_every`
        # was: nothing downstream has to ask which schedule a run used to find
        # the number it wants. `val_history` is the extra, not the substitute.
        "validation": validate(
            cfg, adapter, split, cache, manifest, severity_head, val_sets
        ),
        "val_history": val_history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Written to the run summary, not logged as a step: these are held-out
    # numbers for the finished adapter, and the runs table is where they are
    # compared across a sweep.
    tracker.summary({"validation": summary["validation"], "steps": step})
    tracker.finish()
    return summary


@torch.no_grad()
def _alignment(cfg, adapter, split, cache, manifest, epochs, severity_head=None) -> dict:
    """Per-epoch alignment and detection metrics over one cache. The
    measurement, factored out so held-out degradations and held-out images run
    identically -- if the two were computed by separate code they could drift
    and the comparison between them would stop meaning anything.

    Reports three logits per row, all through the SAME frozen head: `degraded`
    (the input, what the detector scores without GRACE), `adapted` (the adapter's
    output) and `clean` (the cached clean view). One number is uninterpretable
    on its own -- `auc_adapted` alone cannot say whether the adapter helped, and
    `clean` is the ceiling it is being pulled toward, so all three travel
    together.
    """
    device = next(split.parameters()).device
    out = {}
    loader_cfg = _cache_loader_cfg(cfg)
    for epoch in epochs:
        loader = build_loader(
            loader_cfg, cache, manifest, None, epoch, shuffle=False,
            with_taps=adapter.reads_taps,
        )
        cos, gate = [], []
        logits = {"degraded": [], "adapted": [], "clean": []}
        labels = []
        for batch in loader:
            f_deg = _to_float(batch["f_deg"], device)
            f_clean = _to_float(batch["f_clean"], device)
            sev = batch["severity"].to(device).float()
            if severity_head is not None:
                sev = severity_head(f_deg)
            taps = _to_float(batch["taps_deg"], device) if adapter.reads_taps else None
            f_adapted = adapter(f_deg, severity=sev, taps=taps)
            cos.append(
                torch.nn.functional.cosine_similarity(
                    f_adapted.flatten(1), f_clean.flatten(1), dim=1
                ).mean().item()
            )
            gate.append(float(adapter.gate().mean().detach()))
            logits["adapted"].append(split.head(f_adapted).cpu().numpy())
            logits["degraded"].append(split.head(f_deg).cpu().numpy())
            logits["clean"].append(split.head(f_clean).cpu().numpy())
            labels.append(batch["label"].numpy())
        row = {
            "cosine_to_clean": float(np.mean(cos)),
            "gate": float(np.mean(gate)),
        }
        row.update(_detection(np.concatenate(labels),
                              {k: np.concatenate(v) for k, v in logits.items()}))
        out[f"epoch_{epoch}"] = row
    return out


def _detection(y: np.ndarray, logits: dict) -> dict:
    """AUC, accuracy, F1 and retention per view -- through the eval harness's own
    metric functions, not a second implementation of them.

    `pipeline.eval.metrics` is imported rather than reproduced so an in-loop
    number and a reported one cannot drift apart. That includes its threshold
    rule: the operating point is picked on the CLEAN view (max F1) and applied
    unchanged to the degraded and adapted views. A fixed threshold is what
    exposes calibration drift, which AUC hides -- a detector can hold its
    ranking while every degraded row slides to one side of the boundary. It is
    also stable along a training curve here, because the clean view is the same
    rows through the same frozen head at every checkpoint.

    `retention` is `(auc_adapted - 0.5) / (auc_clean - 0.5)`: the fraction of
    chance-corrected clean skill the adapter recovers, so 1.0 is the ceiling and
    0.0 is chance. It is the in-loop echo of the harness's headline number, NOT
    a substitute -- the harness scores the eval split through
    `grace.detectors.adapted`, and that is what gets reported.
    """
    if len(np.unique(y)) < 2:
        return {"note": "one class only; AUC undefined on this cache"}
    thr = threshold_from_clean(logits["clean"], y)
    out = {"threshold": float(thr)}
    for view, score in logits.items():
        errors = error_breakdown(score, y, thr)
        out[f"auc_{view}"] = roc_auc(score, y)
        out[f"acc_{view}"] = errors.accuracy
        out[f"f1_{view}"] = errors.f1
    out["retention"] = retention(out["auc_adapted"], out["auc_clean"])
    return out


@torch.no_grad()
def validate(cfg, adapter, split, cache, manifest, severity_head=None, val_sets=None,
             epoch=None) -> dict:
    """Two held-out axes, reported separately because they answer different
    questions and a single number would hide which one failed.

    `held_out_degradations` -- the original axis (see schedule.val_epochs). The
    val-NUMBERED epochs (from 10000) of the TRAINING cache: unseen corruptions
    over images that were trained on, so it cannot speak to generalization
    across images. Always both val epochs, mid-run and at the end.

    `held_out_images/<name>` -- whole datasets the adapter never saw, rendered to
    their own cache roots. The images are what is held out here, so the
    degradation draw does not also have to be: a training-numbered epoch is the
    right corruption to score, it is simply applied to rows the adapter never
    saw.

    Which of those epochs get scored is what `epoch` selects:

        epoch=e     -- score epoch e alone, the draw training just finished.
                      One pass per val cache instead of fourteen, and every
                      mid-run point is a like-for-like read of the same axis.
        epoch=None  -- score every rendered epoch. The end-of-run pass, where
                      the full sweep is paid once and `validation` stays
                      comparable to every summary.json written before this.

    Each row carries the alignment metrics (cosine to clean, gate) and the
    detection metrics (`auc_*`, `acc_*` for the degraded,
    adapted and clean views, plus `retention`). The detection numbers are the
    in-loop echo of the eval harness, not a replacement for it: the harness
    scores the eval split through `grace.detectors.adapted`, and that is the
    number that gets reported.
    """
    out = {}
    held = [e for e in cache.epochs() if e >= min(val_epochs(1))]
    out["held_out_degradations"] = (
        _alignment(cfg, adapter, split, cache, manifest, held, severity_head)
        if held
        else {"note": "no validation epochs rendered"}
    )
    for name, val_cache, val_manifest in list(val_sets or []):
        rendered = tuple(val_cache.epochs())
        chosen = list(rendered) if epoch is None else [e for e in (epoch,) if e in rendered]
        out[f"held_out_images/{name}"] = (
            _alignment(cfg, adapter, split, val_cache, val_manifest, chosen, severity_head)
            if chosen
            else {"note": f"epoch {epoch} is not rendered under this val cache"}
        )
    return out


def train_discrepancy(cfg, split, manifest) -> dict:
    """Stage 2. Adapter frozen; only the auxiliary head and β train.

    Reports the fused AUC *and* the auxiliary head's standalone AUC. The second
    is the one that matters for experiment E4: run this against a series of
    stage-1 checkpoints and the trend answers whether restoring features destroys
    the drift signal.
    """
    seed_everything(cfg.seed)
    device = next(split.parameters()).device
    spec = split.feature_spec

    # The checkpoint decides whether this is a ladder, not the stage-2 config:
    # stage 2 never rebuilds the adapter, it scores the one stage 1 shipped.
    # `load_adapter` refuses a plain checkpoint against a tapping split and vice
    # versa, so the cache check below can trust `reads_taps`.
    tap_spec = split.tap_spec()
    adapter = load_adapter(cfg.adapter_checkpoint, spec, tap_spec).to(device).eval()
    adapter.requires_grad_(False)

    expect = _expect_spec(
        split, tap_spec if adapter.reads_taps else None, cfg.crop.fingerprint()
    )
    cache = FeatureCache(cfg.cache_dir, expect=expect)
    val_sets = _load_val_sets(cfg, spec, expect)

    # The checkpoint decides the tap count, the config only asks for it. A
    # `use_taps` run against a plain adapter would otherwise train a head whose
    # per-block inputs are all zeros and report it as a null result.
    if cfg.discrepancy.use_taps and not adapter.reads_taps:
        raise ValueError(
            f"discrepancy.use_taps is set but {cfg.adapter_checkpoint} is a plain "
            f"adapter, which computes no per-tap read. Point --adapter at a ladder "
            f"checkpoint, or unset use_taps."
        )
    n_taps = adapter.n_taps if cfg.discrepancy.use_taps else 0
    fused = FusedHead(build_discrepancy_head(spec, cfg.discrepancy, n_taps)).to(device)
    opt = torch.optim.AdamW(fused.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # `adapter_checkpoint` in the config is what makes an E4 sweep readable: the
    # runs table then sorts stage-2 runs by which stage-1 checkpoint they were
    # trained against, which is the x-axis of the erasure figure.
    tracker = build_tracker(
        cfg.wandb, run_id=cfg.run_id, job_type="stage2",
        config={**flatten_config(cfg), "detector_name": split.name,
                "feature_layout": spec.layout, "feature_dim": spec.dim},
    )

    train_epochs = [e for e in cache.epochs() if e < min(val_epochs(1))][: cfg.epochs]
    held = [e for e in cache.epochs() if e >= min(val_epochs(1))]

    loader_cfg = _cache_loader_cfg(cfg)

    step = 0
    for epoch in train_epochs:
        loader = build_loader(
            loader_cfg, cache, manifest, None, epoch, with_taps=adapter.reads_taps
        )
        for batch in tqdm(loader, desc=f"disc epoch {epoch}", leave=False):
            split.assert_frozen()
            f_deg = _to_float(batch["f_deg"], device)
            sev = batch["severity"].to(device).float()
            taps = _to_float(batch["taps_deg"], device) if adapter.reads_taps else None
            with torch.no_grad():
                f_adapted = adapter(f_deg, severity=sev, taps=taps)
                delta = f_adapted - f_deg
                logit_main = split.head(f_adapted)
                tap_drift = adapter.tap_drift(taps) if n_taps else None
            labels = batch["label"].to(device)
            aux_logit = fused.aux(delta, sev, tap_drift)
            bce_fused = supervised_bce(logit_main + fused.beta * aux_logit, labels)
            # The aux head's own objective. See DiscrepancyConfig.lam_aux: with
            # beta starting at 0 the fused term hands it exactly zero gradient,
            # so without this the branch bootstraps off its own random init and
            # the learned sign is arbitrary.
            bce_aux = (
                supervised_bce(aux_logit, labels)
                if cfg.discrepancy.lam_aux > 0
                else torch.zeros((), device=device)
            )
            loss = bce_fused + cfg.discrepancy.lam_aux * bce_aux
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % cfg.log_every == 0:
                tracker.log(
                    {"train/bce": float(bce_fused.detach()),
                     "train/bce_aux": float(bce_aux.detach()),
                     "train/loss": float(loss.detach()),
                     "train/beta": float(fused.beta.detach()),
                     "train/epoch": epoch},
                    step=step,
                )
            step += 1

    out_dir = Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": fused.state_dict(),
            "feature_spec": spec.to_dict(),
            "discrepancy_cfg": vars(cfg.discrepancy),
            # Read off the built head, not off the config: `use_taps` is an
            # intent and the width the weights were trained at is the fact. A
            # checkpoint has to rebuild into the same shape with no reference to
            # the run -- same rule as `save_adapter`'s tap payload.
            "n_taps": fused.aux.n_taps,
            "adapter_checkpoint": cfg.adapter_checkpoint,
        },
        out_dir / "discrepancy.pt",
    )

    # Both axes, named as stage 1 names them. `held_out_degradations` alone is
    # not enough for E4: the base head sits at ~0.9999 AUC there, so `fused`
    # cannot exceed `main` by more than rounding and the branch looks inert
    # whatever it learned. The held-out IMAGE sets are where the main head has
    # room (~0.85 on the hard split), and therefore the only place a
    # discrepancy signal could show up as a gain.
    validation = {
        "held_out_degradations": _score_discrepancy(
            fused, adapter, split, cache, manifest, held or train_epochs, cfg, device
        )
    }
    for name, val_cache, val_manifest in val_sets:
        validation[f"held_out_images/{name}"] = _score_discrepancy(
            fused, adapter, split, val_cache, val_manifest,
            list(val_cache.epochs()), cfg, device,
        )

    summary = {
        "run_id": cfg.run_id,
        "adapter_checkpoint": cfg.adapter_checkpoint,
        "beta": float(fused.beta.detach()),
        "validation": validation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # `auc_aux` is the E4 number: track it across a sweep over stage-1
    # checkpoints and a falling curve is the erasure result.
    tracker.summary({"validation": summary["validation"], "beta": summary["beta"]})
    tracker.finish()
    return summary


@torch.no_grad()
def _score_discrepancy(fused, adapter, split, cache, manifest, epochs, cfg, device) -> dict:
    """Main / auxiliary / fused AUC on held-out degradations.

    Three numbers because the interesting result is the *relationship* between
    them: aux alone above chance means drift carries signal; fused above main
    means it carries signal the main head was not already using.
    """
    loader_cfg = _cache_loader_cfg(cfg)
    out = {}
    for epoch in epochs:
        main, aux, fuse, labels = [], [], [], []
        for batch in build_loader(
            loader_cfg, cache, manifest, None, epoch, shuffle=False,
            with_taps=adapter.reads_taps,
        ):
            f_deg = _to_float(batch["f_deg"], device)
            sev = batch["severity"].to(device).float()
            taps = _to_float(batch["taps_deg"], device) if adapter.reads_taps else None
            delta = adapter(f_deg, severity=sev, taps=taps) - f_deg
            m = split.head(f_deg + delta)
            tap_drift = adapter.tap_drift(taps) if fused.aux.n_taps else None
            a = fused.aux(delta, sev, tap_drift)
            main.append(m.cpu().numpy())
            aux.append(a.cpu().numpy())
            fuse.append((m + fused.beta * a).cpu().numpy())
            labels.append(batch["label"].numpy())
        y = np.concatenate(labels)
        if len(np.unique(y)) < 2:
            continue
        out[f"epoch_{epoch}"] = {
            "auc_main": float(roc_auc_score(y, np.concatenate(main))),
            "auc_aux": float(roc_auc_score(y, np.concatenate(aux))),
            "auc_fused": float(roc_auc_score(y, np.concatenate(fuse))),
        }
    return out
