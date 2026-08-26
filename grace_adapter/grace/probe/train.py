"""Fit the MLP head on CLEAN trunk features. Once, before anything else.

Two passes over the data and a 400k-parameter fit:

    1. one trunk forward per image, ever -- the trunk is frozen and the images
       are not degraded, so the features are constant and are held in memory
    2. AdamW on the head against those features, for as many epochs as wanted at
       no further backbone cost

which is the same argument `grace.cache` makes at a larger scale, and the reason
stage 0 costs seconds rather than being a training run with a checkpoint policy.

**Clean images only, and no augmentation.** This is the premise of the whole
project, not a shortcut. GRACE's claim is about a detector fit on clean data
whose accuracy collapses under JPEG and blur; a head trained with degradation
augmentation would have partly solved that problem already, and every retention
number downstream would be measuring the augmentation instead of the adapter. If
you want that arm, it is a separate detector config and a separate baseline, not
a flag here.

Model selection is on held-out **images**, by AUC. With a PoC-sized manifest a
768-in / 512-hidden head reaches training AUC 1.0 within a few epochs, and the
last epoch is not the one to ship: an overfit head has a near-arbitrary Jacobian,
and `grace.train.weighting` differentiates through it to decide where the adapter
spends its capacity.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from grace.train.tracker import build_tracker, flatten_config
from pipeline.data.dataset import AIGCDataset, collate
from pipeline.detectors.dinov3 import ProbeHead
from pipeline.utils.seeding import seed_everything


@torch.no_grad()
def extract_features(split, manifest, batch_size=32, num_workers=4, device=None):
    """(N, D) clean trunk features and (N,) labels, in manifest order.

    Deliberately reuses `AIGCDataset` with `condition=None` -- the same decode
    path, the same preprocessing and the same row order the cache writer uses
    for its clean view. A probe fit on features extracted any other way would be
    fit on a feature space subtly unlike the one GRACE corrects toward.
    """
    split.assert_frozen()
    device = device or next(split.parameters()).device
    loader = DataLoader(
        AIGCDataset(manifest, preprocess=split.preprocess_fn()),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate,
    )
    feats, labels = [], []
    for batch, metas in tqdm(loader, desc="clean features", leave=False):
        feats.append(split.trunk(batch.to(device)).float().cpu())
        labels.extend(int(m["label"]) for m in metas)
    return torch.cat(feats), torch.tensor(labels, dtype=torch.float32)


def _auc(logits: torch.Tensor, y: torch.Tensor) -> float:
    """NaN when a split is single-class -- possible on a tiny PoC manifest, and
    better said out loud than reported as 0.5."""
    y = y.numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, logits.numpy()))


def train_probe(cfg, split, manifest, val_manifest=None) -> dict:
    """Fit and save the head. Returns the summary written next to it.

    The head trained here is a fresh `ProbeHead`, not `split.detector.head_module`:
    the detector arrives frozen from `build_detector`, and un-freezing a module
    inside it mid-run is exactly the contamination `assert_frozen` exists to
    catch. The trunk stays frozen throughout and is asserted so.
    """
    seed_everything(cfg.seed)
    device = next(split.parameters()).device
    spec = split.feature_spec

    f_train, y_train = extract_features(
        split, manifest, cfg.batch_size, cfg.num_workers, device
    )
    f_val, y_val = (
        extract_features(split, val_manifest, cfg.batch_size, cfg.num_workers, device)
        if val_manifest is not None
        else (None, None)
    )

    head = ProbeHead(spec.dim, cfg.hidden, cfg.n_layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    # Class imbalance is a property of the manifest, not of the method: SID's
    # two fake classes against one real class give a 2:1 pull if `limit` is
    # per-class-name rather than per-label.
    pos = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - pos) / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(
        TensorDataset(f_train, y_train), batch_size=cfg.head_batch_size, shuffle=True
    )

    # Per epoch rather than per step: the whole fit is a few hundred steps on
    # features that are already in memory, and the epoch is the unit model
    # selection happens on.
    tracker = build_tracker(
        cfg.wandb, run_id=cfg.run_id, job_type="stage0_probe",
        config={**flatten_config(cfg), "backbone_id": split.detector.backbone_id,
                "pool": split.detector.pool, "feature_dim": spec.dim,
                "input_mode": getattr(split.detector, "input_mode", "resize"),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)) if f_val is not None else 0,
                "head_params": sum(p.numel() for p in head.parameters())},
    )

    best = {"auc": -1.0, "state": None, "epoch": -1}
    history = []
    for epoch in range(cfg.epochs):
        split.assert_frozen()
        head.train()
        for f, y in loader:
            loss = criterion(head(f.to(device)), y.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()

        head.eval()
        with torch.no_grad():
            row = {"epoch": epoch, "loss": float(loss), "auc_train": _auc(head(f_train.to(device)).cpu(), y_train)}
            if f_val is not None:
                logits = head(f_val.to(device)).cpu()
                row["auc_val"] = _auc(logits, y_val)
                row["acc_val"] = float(((logits > 0).float() == y_val).float().mean())
        history.append(row)
        tracker.log({f"probe/{k}": v for k, v in row.items() if k != "epoch"}, step=epoch)

        # No validation manifest -> select on the last epoch and say so in the
        # summary, rather than silently selecting on training AUC.
        score = row.get("auc_val", float("-inf"))
        if f_val is None or (score == score and score >= best["auc"]):
            best = {
                "auc": score,
                "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "epoch": epoch,
            }

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best["state"] or head.state_dict(),
            # Everything `DINOv3MLPDetector.__init__` needs to rebuild this head
            # and to refuse it if it does not belong to the trunk being loaded.
            "backbone_id": split.detector.backbone_id,
            "pool": split.detector.pool,
            "input_mode": getattr(split.detector, "input_mode", "resize"),
            "feature_dim": spec.dim,
            "hidden": cfg.hidden,
            "n_layers": cfg.n_layers,
            "dropout": cfg.dropout,
            "run_id": cfg.run_id,
            "selected_epoch": best["epoch"],
        },
        out,
    )

    summary = {
        "run_id": cfg.run_id,
        "checkpoint": str(out),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)) if f_val is not None else 0,
        "selection": "val_auc" if f_val is not None else "last_epoch",
        "selected_epoch": best["epoch"],
        "best_auc_val": best["auc"] if f_val is not None else None,
        "history": history,
    }
    import json

    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The selected epoch's numbers, not the last epoch's -- the shipped head is
    # the selected one, and the runs table should say what it scored.
    tracker.summary({
        "selected_epoch": best["epoch"],
        "best_auc_val": best["auc"] if f_val is not None else None,
        **{k: v for k, v in history[best["epoch"]].items() if k != "epoch"},
    })
    tracker.finish()
    return summary
