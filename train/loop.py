"""The training stages.

**Stage 1 -- the adapter.** Label-free. In `source: cache` mode a step is:

    f_deg, f_clean = batch                     # two memmap reads
    f_adapted      = adapter(f_deg.float())    # ~2M params
    loss           = total_loss(...)           # + one frozen head for head_kl
    loss.backward(); opt.step(); ema.update()

No image decode, no augmentation, no trunk. A full run costs minutes, which is
the point of pre-rendering: the seed sweep, the geometry grid and the weight soup
that normally get cut for time become affordable, and any ablation can be re-run
whenever a question about it arises.

**Stage 2 -- the frequency enricher.** Supervised, adapter frozen. Reads the
image again in a basis the trunk's resize threw away (see `train_enrich` below).

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

from train.cache.reader import FeatureCache
from train.cache.spec import CacheSpec
from train.cache.schedule import val_epochs
from grace_adapter.models.factory import (
    build_adapter, build_severity_head, load_adapter, save_adapter,
)
from freq_branch.models.factory import build_enricher, save_enricher
from freq_branch.models.frequency import EnricherFusedLogit
from train import diagnostics as D
from train.data import build_loader
from train.ema import EMA
from train.losses import orthogonality_loss, supervised_bce, total_loss
from train.tracker import build_tracker, flatten, flatten_config
from train.weighting import head_gradient
from load_data.config import load_dataset_config
from load_data.manifest import load_manifest
from eval.metrics import (
    error_breakdown, retention, roc_auc, threshold_from_clean,
)
from common.seeding import seed_everything


def cosine_with_warmup(opt, warmup: int, total: int) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(opt, fn)


def _warmup_lambda(warmup: int, total: int):
    """A `step -> lr multiplier` for ONE param group, or a constant 1.0 when the
    group opts out.

    The shape is `cosine_with_warmup`'s -- linear ramp from ~1/warmup to 1, then
    a cosine decay to zero over the run -- but exposed per group so two modules
    can warm up at different rates from different base LRs. `warmup <= 0` means
    "no schedule", returning exactly the constant-LR multiplier that reproduces
    the stage's pre-scheduler behaviour rather than a decay it never asked for.
    """
    if warmup <= 0:
        return lambda step: 1.0

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def differential_warmup(opt, warmups: list[int], total: int) -> LambdaLR:
    """`LambdaLR` over the optimizer's param groups, one `warmups` entry each.

    `LambdaLR` with a list of functions applies the i-th to the i-th group, so
    the base LR each group was created with (enricher `lr`, head `head_lr`) is
    preserved and only its multiplier differs. A group with a warmup of 0 keeps a
    constant LR. The caller is responsible for a length match, which is
    `len(opt.param_groups)`.
    """
    return LambdaLR(opt, [_warmup_lambda(w, total) for w in warmups])


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


def _expect_spec(split, crop_sha: str = "") -> CacheSpec:
    """What this run needs a cache to be, for `CacheSpec.assert_compatible`.

    Only the parts that vary per run go in: the feature layout and the crop. The
    four fingerprints are left blank because the *cache* is the authority on
    those -- a run does not know the manifest hash it wants, it knows the split
    it is training against.

    `crop_sha` is the exception to "the cache is the authority", and it is passed
    rather than blank because the run genuinely does know which window protocol
    it wants: its own `crop:` block determines what `source: live` will draw and
    what the clean targets have to be windows of. A blank here would make every
    cropped cache readable by a whole-image run.
    """
    return CacheSpec(detector=split.name, feature=split.feature_spec, n=0, crop_sha=crop_sha)


def _load_val_sets(cfg, spec, expect: CacheSpec | None = None) -> list:
    """Held-out IMAGES, each from its own rendered cache root.

    Loaded up front so a missing or mis-specified val cache fails now rather
    than after the full training run, at the one moment its result is wanted.

    Shared by both stages: stage 2 scores the same held-out images as stage 1,
    against the same integrity checks, so the two are read off the same axis
    rather than two that merely look alike.
    """
    val_sets = []
    for ds_path, cache_dir in zip(
        getattr(cfg, "val_datasets", []) or [], getattr(cfg, "val_cache_dirs", []) or []
    ):
        ds_cfg = load_dataset_config(ds_path)
        val_cache = FeatureCache(cache_dir, expect=expect)
        val_manifest = load_manifest(ds_cfg.manifest, ds_cfg.split)
        if len(val_manifest) != val_cache.spec.n:
            raise RuntimeError(
                f"{cache_dir} holds {val_cache.spec.n} rows but {ds_path} selects "
                f"{len(val_manifest)}. The cache was rendered from a different "
                f"manifest -- re-render it with scripts/main/build_cache.py."
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

    `decay_gate: false` puts `gate_logit` in a group with no decay and leaves
    every other parameter where it was, so the arm differs from its control in
    the gate alone. Default is `true`, which reproduces the single group the
    runs before this were trained under, byte for byte.
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

    expect = _expect_spec(split, cfg.crop.fingerprint())
    cache = FeatureCache(cfg.cache_dir, expect=expect)

    val_sets = _load_val_sets(cfg, spec, expect)

    adapter = build_adapter(spec, cfg.adapter).to(device)
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
            # Only reaches `source: live`, and there it must be the crop the
            # cache's clean view was rendered under: `f_clean` is the target for
            # whatever window `image` shows. `crop_sha` on the cache is what
            # catches a run that gets this wrong.
            crop=cfg.crop.build(),
        )
        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            split.assert_frozen()

            f_clean = _to_float(batch["f_clean"], device)
            if cfg.source == "live":
                with torch.no_grad():
                    f_deg = split.trunk(batch["image"].to(device)).float()
            else:
                f_deg = _to_float(batch["f_deg"], device)
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
            f_adapted = adapter(f_deg, severity=sev_in)

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
                # read. The total is free -- clipping computes it anyway.
                terms["grad_norm"] = float(total_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            ema.update(adapter)

            if logging_step:
                with torch.no_grad():
                    terms["gate"] = float(adapter.gate().mean().detach())
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

        # Intermediate checkpoints, written at the same cadence as `val_every`,
        # so each can be paired with the numbers it scored.
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
    ema_adapter = build_adapter(spec, cfg.adapter)
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
        loader = build_loader(loader_cfg, cache, manifest, None, epoch, shuffle=False)
        cos, gate = [], []
        logits = {"degraded": [], "adapted": [], "clean": []}
        labels = []
        for batch in loader:
            f_deg = _to_float(batch["f_deg"], device)
            f_clean = _to_float(batch["f_clean"], device)
            sev = batch["severity"].to(device).float()
            if severity_head is not None:
                sev = severity_head(f_deg)
            f_adapted = adapter(f_deg, severity=sev)
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

    `eval.metrics` is imported rather than reproduced so an in-loop
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
    `grace_adapter.detectors.adapted`, and that is what gets reported.
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
    scores the eval split through `grace_adapter.detectors.adapted`, and that is the
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


# =============================================================================
# Stage 2, the frequency branch. Reads the image again, in a basis the trunk's
# resize threw away, against the same frozen adapter stage 1 shipped.
# =============================================================================


def train_enrich(cfg, split, manifest) -> dict:
    """Stage 2 for the frequency branch. Supervised; the adapter stays frozen.

    Reads per step, all from one cache: `f_deg`, `freq_deg`, `label`,
    `severity`. `f_clean` is not read -- there is no restoration target here.
    The objective is

        BCE(head(fused) + beta * aux(delta), label)          # fused logit
      + lam_aux * BCE(aux(delta), label)                     # standalone DCT read
      + lam_orth * orthogonality(delta, f_corrected)         # complementarity

    where `fused = f_corrected + sum_b gate_b * expert_b(...)` is the feature
    enrichment and `delta = sum_b expert_b(tokens, f, pos)` is the DCT branch's
    UN-GATED read. The prediction is the first line -- the main head's read of
    the fused features plus a learned scalar beta times the aux logit. beta
    starts at 0, so at step 0 the
    prediction is exactly `head(fused)` and the identity survives; a beta that
    leaves zero is direct evidence the frequency signal has no decision value the
    fused objective can use, independent of whether the head can read the
    enriched FEATURES. `fuse_aux_logit: false` drops the beta term and scores
    `head(fused)` alone, which is the control this arm exists to beat.

    The aux and orth terms both read the UN-GATED expert sum, on purpose: a loss
    on the gated sum could be satisfied by closing a gate, and a closed gate
    teaches the experts nothing. The aux head forces the frequency read to be
    label-predictive on its own -- an independently readable forensic signature
    -- and the orthogonality term forces it to point where the spatial feature
    does not already span. The anchor term (`lam_anchor`) is removed from the
    objective: it is off by default and the field survives only as the E15
    ablation.

    THE IDENTITY IS MEASURED, NOT ASSUMED. Validation runs once before the first
    optimizer step, and at that point every expert's output projection is still
    zero (so `fused` equals `f_corrected`), and beta is 0, so the prediction is
    exactly the `+grace` arm's and `auc_fused` must equal `auc_corrected`. It is
    recorded as `validation.step_0`, and it is the cheapest possible check that
    the module is wired where it claims to be.

    E14 ships as `finetune_head`. Frozen is the default and the one whose
    provenance is worth something; the fine-tuned arm exists because a frozen
    head can only read this whole cross-attention module as a scalar logit shift.
    """
    seed_everything(cfg.seed)
    device = next(split.parameters()).device
    spec = split.feature_spec

    # This stage never rebuilds the adapter, it enriches the one stage 1 shipped.
    adapter = load_adapter(cfg.adapter_checkpoint, spec).to(device).eval()
    adapter.requires_grad_(False)
    severity_head = _load_severity_head(cfg.adapter_checkpoint, adapter, spec, device)

    freq_spec = cfg.freq.feature()
    expect = _expect_spec(split, cfg.crop.fingerprint())
    cache = FeatureCache(cfg.cache_dir, expect=expect)
    cache.spec.assert_freq_available(freq_spec, cfg.freq.fingerprint())
    val_sets = _load_val_sets(cfg, spec, expect)
    for _, val_cache, _ in val_sets:
        val_cache.spec.assert_freq_available(freq_spec, cfg.freq.fingerprint())

    enricher = build_enricher(
        spec, freq_spec, cfg.enricher, patch=cfg.freq.patch, channels=cfg.freq.channels
    ).to(device)

    # The head is the detector's own, and by default it is not touched. When it
    # is, a COPY is trained: `split.head` stays the frozen reference every other
    # arm is scored with, so the fine-tuned arm cannot quietly change what the
    # baseline means.
    head = _enrich_head(split, cfg, device)
    # The fused-logit module (aux head + beta) is a training scaffold on the DCT
    # branch's own read, so it is trained in the enricher's group and at the
    # enricher's LR -- the same ramp as the module it is shaping, not a schedule
    # of its own. It is built only when one of the terms that read it is on.
    fused_logit = (
        _enrich_fused_logit(spec, cfg, device)
        if (cfg.lam_aux or cfg.lam_orth or cfg.fuse_aux_logit) else None
    )
    enricher_params = list(enricher.parameters())
    if fused_logit is not None:
        enricher_params = enricher_params + list(fused_logit.parameters())
    groups = [{"params": enricher_params, "lr": cfg.lr}]
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": cfg.head_lr})
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_epochs = [e for e in cache.epochs() if e < min(val_epochs(1))][: cfg.epochs]
    held = [e for e in cache.epochs() if e >= min(val_epochs(1))]
    # Differential warmup: each group gets its own `step -> multiplier`, so the
    # arriving-trained head can be held near frozen (its LR ramps from ~1/N)
    # while the enricher trains at full rate, then released to co-adapt at
    # `head_lr`. `0` means constant LR, which is exactly the behaviour every
    # stage-2 run shipped before this field existed.
    sched = None
    if cfg.warmup_steps or (head is not None and cfg.head_warmup_steps):
        warmups = [cfg.warmup_steps]
        if head is not None:
            warmups.append(cfg.head_warmup_steps)
        total = (len(manifest) // cfg.batch_size) * len(train_epochs)
        sched = differential_warmup(opt, warmups, total)

    def logit_of(f):
        return (head if head is not None else split.head)(f)

    tracker = build_tracker(
        cfg.wandb, run_id=cfg.run_id, job_type="stage2-enrich",
        config={**flatten_config(cfg), "detector_name": split.name,
                "feature_layout": spec.layout, "feature_dim": spec.dim,
                "freq_shape": list(freq_spec.shape)},
    )

    loader_cfg = _cache_loader_cfg(cfg)

    # Before any step, with the enricher still exactly the identity. Everything
    # after this is read as a delta from it.
    validation = {
        "step_0": _score_enrich(
            enricher, adapter, severity_head, logit_of, fused_logit, cache, manifest,
            held or train_epochs, cfg, device,
        )
    }

    step = 0
    for epoch in train_epochs:
        loader = build_loader(loader_cfg, cache, manifest, None, epoch, with_freq=True)
        for batch in tqdm(loader, desc=f"enrich epoch {epoch}", leave=False):
            split.assert_frozen()
            f_deg = _to_float(batch["f_deg"], device)
            freq = _to_float(batch["freq_deg"], device)
            sev = batch["severity"].to(device).float()
            with torch.no_grad():
                # Stage 1's PREDICTED severity, not the recipe's recorded one: at
                # evaluation time no recipe is known, so conditioning on the
                # recorded scalar here and on a prediction there would train and
                # score two different models. `sev` survives only as a diagnostic.
                pred_sev = severity_head(f_deg) if severity_head is not None else None
                f_corrected = adapter(f_deg, severity=pred_sev)
            labels = batch["label"].to(device)

            fused = enricher(f_corrected, freq, pred_sev)
            # The main objective reads the FUSED LOGIT: `head(fused)` plus, when
            # `fuse_aux_logit`, a learned scalar beta times the aux logit. beta
            # starts at 0, so early steps are exactly the feature-fusion loss.
            # The DCT branch's own read, before the gate, is what the aux and
            # orth terms read -- a loss on the gated sum could be satisfied by
            # closing a gate, and a closed gate teaches the experts nothing. Only
            # one `update()` call even when several terms use it.
            delta_freq = None
            if fused_logit is not None:
                delta_freq = enricher.update(f_corrected, freq)
            if fused_logit is not None and cfg.fuse_aux_logit:
                bce = supervised_bce(fused_logit(logit_of(fused), delta_freq), labels)
            else:
                bce = supervised_bce(logit_of(fused), labels)
            bce_aux = torch.zeros((), device=device)
            orth = torch.zeros((), device=device)
            if delta_freq is not None:
                if cfg.lam_aux:
                    bce_aux = supervised_bce(fused_logit.aux(delta_freq), labels)
                if cfg.lam_orth:
                    if cfg.orth_target == "decision":
                        # The head's per-sample decision direction. The orth term
                        # normalizes its inputs, so passing j is exactly
                        # orthogonalizing Δ against the decision-weighted
                        # projection of f (proj_j(f) ∝ j): Δ may not re-state
                        # the part of the spatial read the head weighs, but is
                        # free everywhere else. `head_gradient` runs on its own
                        # graph and detaches, so this never leaks into the
                        # enricher's or the head's.
                        orth = orthogonality_loss(
                            delta_freq, head_gradient(logit_of, f_corrected)
                        )
                    else:
                        orth = orthogonality_loss(delta_freq, f_corrected)
            # The E15 anchor, off by default. Kept behind `lam_anchor > 0` so a
            # default run never pays for a term it does not train on.
            anchor = torch.zeros((), device=device)
            if cfg.lam_anchor:
                anchor = (fused - f_corrected).norm(dim=-1).mean()
            loss = (bce + cfg.lam_aux * bce_aux + cfg.lam_orth * orth
                    + cfg.lam_anchor * anchor)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                gates = enricher.gates().detach()
                log = {
                    "train/bce": float(bce.detach()),
                    "train/bce_aux": float(bce_aux.detach()),
                    "train/orth": float(orth.detach()),
                    "train/anchor": float(anchor.detach()),
                    "train/loss": float(loss.detach()),
                    "train/severity_recipe": float(sev.mean()),
                    "train/epoch": epoch,
                    # Effective LRs AFTER this step's scheduler update: the
                    # head's is the one a differential warmup is about, so it
                    # has to be on the dashboard and not just in the config.
                    "train/lr_enricher": float(opt.param_groups[0]["lr"]),
                    **{f"train/gate_band{b}": float(gates[b].mean())
                       for b in range(enricher.n_bands)},
                }
                if fused_logit is not None and cfg.fuse_aux_logit:
                    log["train/beta"] = float(fused_logit.beta.detach())
                if head is not None:
                    log["train/lr_head"] = float(opt.param_groups[1]["lr"])
                tracker.log(log, step=step)
            step += 1

    out_dir = Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_enricher(
        out_dir / "enricher.pt", enricher, spec, freq_spec, cfg.enricher, cfg.freq,
        extra={
            "adapter_checkpoint": cfg.adapter_checkpoint,
            # The fine-tuned head travels WITH the enricher it was fitted
            # alongside. The two are only meaningful together, and a detector
            # config naming one without the other would score a head against
            # features it never saw -- the exact failure `_assert_head_matches`
            # exists to stop, one stage later.
            "head_state_dict": None if head is None else head.state_dict(),
            "finetune_head": head is not None,
            # The fused logit (aux head + beta) travels too, for the same reason:
            # when `fuse_aux_logit`, the prediction is `head(fused) + beta * aux`,
            # so the detector scoring the enricher must reconstruct the module
            # that beta multiplies. `aux_cfg` is the geometry that rebuilding
            # needs (input is always `spec.dim`, only `hidden` varies).
            "fused_logit_state_dict": (
                None if fused_logit is None or not cfg.fuse_aux_logit
                else fused_logit.state_dict()
            ),
            "aux_cfg": {"hidden": cfg.aux_hidden},
            "fuse_aux_logit": cfg.fuse_aux_logit,
        },
    )

    validation["held_out_degradations"] = _score_enrich(
        enricher, adapter, severity_head, logit_of, fused_logit, cache, manifest,
        held or train_epochs, cfg, device,
    )
    for name, val_cache, val_manifest in val_sets:
        validation[f"held_out_images/{name}"] = _score_enrich(
            enricher, adapter, severity_head, logit_of, fused_logit, val_cache,
            val_manifest, list(val_cache.epochs()), cfg, device,
        )

    summary = {
        "run_id": cfg.run_id,
        "adapter_checkpoint": cfg.adapter_checkpoint,
        "finetune_head": head is not None,
        "fuse_aux_logit": cfg.fuse_aux_logit,
        "beta": None if fused_logit is None or not cfg.fuse_aux_logit
                else float(fused_logit.beta.detach()),
        "gates": [float(g.mean()) for g in enricher.gates().detach()],
        "validation": validation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tracker.summary(
        {k: summary[k] for k in ("validation", "gates", "finetune_head",
                                 "fuse_aux_logit", "beta")}
    )
    tracker.finish()
    return summary


def _enrich_head(split, cfg, device):
    """A trainable COPY of the detector's head, or None for the frozen arm.

    `deepcopy` rather than rebuild-and-load: the head's geometry lives in the
    detector config and in its checkpoint, and restating it here would be a third
    place for the three to disagree. The copy is put into train mode -- the
    original is frozen and in eval, so dropout in the source head would otherwise
    stay silently off.
    """
    if not cfg.finetune_head:
        return None
    import copy

    head = copy.deepcopy(split.head_module()).to(device)
    head.requires_grad_(True)
    head.train()
    return head


def _enrich_fused_logit(spec, cfg, device):
    """The aux head + learned beta (logit-level fusion) on the DCT read.

    Built fresh rather than deep-copied: the aux head is a training scaffold that
    shares the enricher's optimizer group and LR. Its `beta` is the scalar that
    decides, at the logit level, whether the frequency signal is used at all --
    the one number whose size is the experiment. Trained in the enricher's group
    so the differential warmup schedules it exactly as it schedules the module it
    is shaping.
    """
    module = EnricherFusedLogit(spec.dim, hidden=cfg.aux_hidden).to(device)
    module.requires_grad_(True)
    module.train()
    return module


def _load_severity_head(checkpoint: str, adapter, spec, device):
    """Stage 1's severity head, if the checkpoint carries one.

    Same rule `grace_adapter.detectors.adapted` follows: without it the FiLM path has no
    input and conditioning silently reverts to the unconditioned gate -- so an
    adapter trained with FiLM must be scored beside the head it was trained with,
    at every later stage.
    """
    if adapter.film is None:
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "severity_state_dict" not in payload:
        return None
    head = build_severity_head(spec).to(device)
    head.load_state_dict(payload["severity_state_dict"])
    return head.eval().requires_grad_(False)


@torch.no_grad()
def _score_enrich(enricher, adapter, severity_head, logit_of, fused_logit, cache,
                  manifest, epochs, cfg, device) -> dict:
    """Corrected / fused AUC per epoch, plus how far the fusion moved.

    `auc_corrected` is the `+grace` arm computed on the same rows in the same
    pass, so the comparison is paired rather than read across two runs -- and at
    step 0 the two columns must agree to the last decimal, which is E10 measured
    instead of asserted.

    `auc_fused` is the actual prediction: `head(fused)` when `fuse_aux_logit` is
    off (or no aux head exists), else `head(fused) + beta * aux(update)`. That is
    the one number the β-fusion arm is about -- whether the scalar can use the
    frequency signal even when the head cannot read the enriched FEATURES.

    `auc_aux` is the aux head scored ALONE on the DCT branch's un-gated read --
    the number that says whether the frequency signatures are readable
    standalone. It is absent (not 0) when the head is off or its logits are
    constant, so a random-init step-0 row does not report a number that means
    nothing.
    """
    was_training = enricher.training
    enricher.eval()
    loader_cfg = _cache_loader_cfg(cfg)
    out = {}
    for epoch in epochs:
        base, fuse, drift, labels, aux_logits = [], [], [], [], []
        for batch in build_loader(
            loader_cfg, cache, manifest, None, epoch, shuffle=False, with_freq=True,
        ):
            f_deg = _to_float(batch["f_deg"], device)
            freq = _to_float(batch["freq_deg"], device)
            sev = severity_head(f_deg) if severity_head is not None else None
            f_corrected = adapter(f_deg, severity=sev)
            fused = enricher(f_corrected, freq, sev)
            base.append(logit_of(f_corrected).cpu().numpy())
            drift.append((fused - f_corrected).norm(dim=-1).cpu().numpy())
            labels.append(batch["label"].numpy())
            if fused_logit is not None:
                update = enricher.update(f_corrected, freq)
                if cfg.fuse_aux_logit:
                    fuse.append(fused_logit(logit_of(fused), update).cpu().numpy())
                else:
                    fuse.append(logit_of(fused).cpu().numpy())
                aux_logits.append(fused_logit.aux(update).cpu().numpy())
            else:
                fuse.append(logit_of(fused).cpu().numpy())
        y = np.concatenate(labels)
        if len(np.unique(y)) < 2:
            continue
        row = {
            "auc_corrected": float(roc_auc_score(y, np.concatenate(base))),
            "auc_fused": float(roc_auc_score(y, np.concatenate(fuse))),
            "enrichment_norm": float(np.concatenate(drift).mean()),
        }
        if len(aux_logits):
            a = np.concatenate(aux_logits)
            if len(np.unique(a)) > 1:
                row["auc_aux"] = float(roc_auc_score(y, a))
        out[f"epoch_{epoch}"] = row
    enricher.train(was_training)
    return out
