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
from grace.cache.schedule import val_epochs
from grace.models.discrepancy import FusedHead
from grace.models.factory import (
    build_adapter, build_discrepancy_head, build_severity_head, load_adapter, save_adapter,
)
from grace.train import diagnostics as D
from grace.train.data import build_loader
from grace.train.ema import EMA
from grace.train.losses import supervised_bce, total_loss
from grace.train.weighting import head_gradient
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
        "num_workers": cfg.num_workers,
    })()


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

    cache = FeatureCache(cfg.cache_dir)
    adapter = build_adapter(spec, cfg.adapter).to(device)
    severity_head = (
        build_severity_head(spec).to(device) if cfg.loss.lam_sev > 0 else None
    )
    params = list(adapter.parameters()) + (
        list(severity_head.parameters()) if severity_head else []
    )
    ema = EMA(adapter, cfg.ema_decay)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    epochs = [e for e in cache.epochs() if e < min(val_epochs(1))][: cfg.epochs]
    if not epochs:
        raise RuntimeError(f"no rendered training epochs under {cfg.cache_dir}")
    steps_per_epoch = len(manifest) // cfg.batch_size
    sched = cosine_with_warmup(opt, cfg.warmup_steps, steps_per_epoch * len(epochs))

    out_dir = Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess = split.preprocess_fn() if cfg.source == "live" else None

    step, history = 0, []
    for epoch in epochs:
        loader = build_loader(cfg, cache, manifest, schedule, epoch, preprocess)
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
            k = cfg.sampling.k_train if adapter.stochastic else 1
            f_adapted = adapter.sample(f_deg, k, severity=sev_in)

            loss, terms = total_loss(
                adapter=adapter, head=split.head,
                f_adapted=f_adapted, f_clean=target, j=j,
                severity_pred=sev_pred, severity_target=sev_target,
                cfg=cfg.loss,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            ema.update(adapter)

            if step % 50 == 0:
                with torch.no_grad():
                    terms["gate"] = float(adapter.gate().mean().detach())
                    if j is not None:
                        terms["cos_decision"] = float(
                            D.decision_alignment(f_adapted[0], f_deg, j).abs().mean()
                        )
                    terms["step"] = step
                    history.append(dict(terms))
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
        "validation": validate(cfg, adapter, split, cache, manifest, severity_head),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def validate(cfg, adapter, split, cache, manifest, severity_head=None) -> dict:
    """Alignment metrics on held-out *degradations* (see schedule.val_epochs).

    Deliberately not AUC: retention is measured by the eval harness, on the eval
    split, through `grace.detectors.adapted`. This is the in-loop signal only --
    cosine to the clean target, mean gate, decision alignment and posterior
    spread, so a run that helps L1 while wrecking L3 is visible before it ends.
    """
    device = next(split.parameters()).device
    held = [e for e in cache.epochs() if e >= min(val_epochs(1))]
    if not held:
        return {"note": "no validation epochs rendered"}

    out = {}
    loader_cfg = _cache_loader_cfg(cfg)
    for epoch in held:
        loader = build_loader(loader_cfg, cache, manifest, None, epoch, shuffle=False)
        cos, spread, gate = [], [], []
        for batch in loader:
            f_deg = _to_float(batch["f_deg"], device)
            f_clean = _to_float(batch["f_clean"], device)
            sev = batch["severity"].to(device).float()
            if severity_head is not None:
                sev = severity_head(f_deg)
            k = cfg.sampling.k_eval if adapter.stochastic else 1
            draws = adapter.sample(f_deg, k, severity=sev)
            cos.append(
                torch.nn.functional.cosine_similarity(
                    draws.mean(0).flatten(1), f_clean.flatten(1), dim=1
                ).mean().item()
            )
            spread.append(D.posterior_spread(torch.stack([split.head(d) for d in draws])))
            gate.append(float(adapter.gate().mean().detach()))
        out[f"epoch_{epoch}"] = {
            "cosine_to_clean": float(np.mean(cos)),
            "posterior_spread": float(np.mean(spread)),
            "gate": float(np.mean(gate)),
        }
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

    cache = FeatureCache(cfg.cache_dir)
    adapter = load_adapter(cfg.adapter_checkpoint, spec).to(device).eval()
    adapter.requires_grad_(False)

    fused = FusedHead(build_discrepancy_head(spec, cfg.discrepancy)).to(device)
    opt = torch.optim.AdamW(fused.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_epochs = [e for e in cache.epochs() if e < min(val_epochs(1))][: cfg.epochs]
    held = [e for e in cache.epochs() if e >= min(val_epochs(1))]

    loader_cfg = _cache_loader_cfg(cfg)

    for epoch in train_epochs:
        loader = build_loader(loader_cfg, cache, manifest, None, epoch)
        for batch in tqdm(loader, desc=f"disc epoch {epoch}", leave=False):
            split.assert_frozen()
            f_deg = _to_float(batch["f_deg"], device)
            sev = batch["severity"].to(device).float()
            with torch.no_grad():
                f_adapted = adapter(f_deg, severity=sev)
                delta = f_adapted - f_deg
                logit_main = split.head(f_adapted)
            loss = supervised_bce(fused(logit_main, delta, sev), batch["label"].to(device))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    out_dir = Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": fused.state_dict(),
            "feature_spec": spec.to_dict(),
            "discrepancy_cfg": vars(cfg.discrepancy),
            "adapter_checkpoint": cfg.adapter_checkpoint,
        },
        out_dir / "discrepancy.pt",
    )

    summary = {
        "run_id": cfg.run_id,
        "adapter_checkpoint": cfg.adapter_checkpoint,
        "beta": float(fused.beta),
        "validation": _score_discrepancy(fused, adapter, split, cache, manifest,
                                         held or train_epochs, cfg, device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
        for batch in build_loader(loader_cfg, cache, manifest, None, epoch, shuffle=False):
            f_deg = _to_float(batch["f_deg"], device)
            sev = batch["severity"].to(device).float()
            delta = adapter(f_deg, severity=sev) - f_deg
            m = split.head(f_deg + delta)
            a = fused.aux(delta, sev)
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
