"""Does the DCT read carry decision info the spatial features already have?

    python scripts/misc/probe_concat.py

Fits two heads on the SAME cached clean features (same window, same images):

    arm A  spatial   head(f)                       dim 768
    arm B  concat    head([f, mean_cells(freq)])   dim 768 + 192

both selected on held-out-image AUC, per the probe premise (clean-only fit, no
degradation augmentation). The `+freq` arm is then ALSO scored on a held-out
DEGRADED view, which is the data the stage-2 enricher actually fuses on.

If B cannot beat A on held-out images, the frequency read is decision-redundant
with the spatial features -- the enricher's premise fails on the very data it
would fuse. The readout here is deliberately the weakest possible one (a global
mean over the 196 cells): it is a NECESSARY condition for the premise, so a
failure is decisive while a success only says the signal survives pooling. The
per-cell structure is the enricher's job, and this probe exists to decide whether
that job has anything to read.

Reads straight from the cache -- no trunk pass, no images decoded -- which is
also what guarantees the spatial and frequency features are of the same window
of the same pixels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from train.cache.reader import FeatureCache
from load_data.config import load_dataset_config
from load_data.manifest import load_manifest
from eval.detectors.dinov3 import ProbeHead
from common.seeding import seed_everything


def _parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="cache_ms/dinov3-wildfake-multiscale")
    p.add_argument("--val-cache-dir", default="cache_ms_val/dinov3-wildfake-multiscale")
    p.add_argument("--train-dataset", default="load_data/configs/datasets/wildfake_train.yaml")
    p.add_argument("--val-dataset", default="load_data/configs/datasets/wildfake_train_val.yaml")
    p.add_argument("--fit-view", default="clean", help="'clean' or 'epoch=0'")
    p.add_argument("--report-epoch", default="epoch=0", help="degraded view for the retention score")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/probe_concat/summary.json")
    return p.parse_args()


@torch.no_grad()
def _read(cache: FeatureCache, manifest, fit_view: str, report_epoch: str,
          chunk: int = 4000) -> dict:
    """(spatial, freq_mean, label) tensors for fit and report views, in row order."""
    n = len(manifest)
    labels = torch.as_tensor(manifest["label"].to_numpy(dtype=np.int64), dtype=torch.float32)
    f_fit = torch.empty((n, cache.spec.feature.shape[0]), dtype=torch.float32)
    fr_fit = torch.empty((n, cache.spec.freq_feature.shape[1]), dtype=torch.float32)
    f_rep = torch.empty_like(f_fit)
    fr_rep = torch.empty_like(fr_fit)
    index = cache.index
    fit_epoch = None if fit_view == "clean" else int(fit_view.split("=")[1])
    rep_epoch = None if report_epoch == "clean" else int(report_epoch.split("=")[1])
    for lo in range(0, n, chunk):
        rows = index[lo : lo + chunk]
        f_fit[lo : lo + chunk] = (
            cache.clean(rows) if fit_epoch is None else cache.degraded(rows, fit_epoch)
        ).float()
        fr_fit[lo : lo + chunk] = (
            cache.clean_freq(rows) if fit_epoch is None else cache.freq(rows, fit_epoch)
        ).float().mean(dim=1)
        f_rep[lo : lo + chunk] = cache.degraded(rows, rep_epoch).float()
        fr_rep[lo : lo + chunk] = cache.freq(rows, rep_epoch).float().mean(dim=1)
    return {
        "f_fit": f_fit, "fr_fit": fr_fit, "f_rep": f_rep, "fr_rep": fr_rep,
        "labels": labels, "fit_view": fit_view, "report_epoch": report_epoch,
    }


def _auc(logits: torch.Tensor, y: torch.Tensor) -> float:
    y = y.numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, logits.numpy()))


def _fit(name: str, feat: torch.Tensor, y: torch.Tensor, val_feat: torch.Tensor,
         y_val: torch.Tensor, epochs: int, hidden: int, seed: int) -> dict:
    """Fit a ProbeHead on (N, D) features; return the best-val-epoch numbers."""
    seed_everything(seed)
    head = ProbeHead(feat.shape[1], hidden, 2, 0.0)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    pos = float(y.sum())
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - pos) / max(pos, 1.0)]))
    loader = DataLoader(TensorDataset(feat, y), batch_size=512, shuffle=True)
    best = {"auc": -1.0, "state": None, "epoch": -1}
    history = []
    for epoch in range(epochs):
        head.train()
        for f, yy in loader:
            loss = crit(head(f), yy)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        head.eval()
        with torch.no_grad():
            auc_train = _auc(head(feat).cpu(), y.cpu())
            auc_val = _auc(head(val_feat).cpu(), y_val.cpu())
        history.append({"epoch": epoch, "auc_train": auc_train, "auc_val": auc_val})
        if auc_val == auc_val and auc_val >= best["auc"]:
            best = {"auc": auc_val, "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, "epoch": epoch}
    return {"name": name, "params": sum(p.numel() for p in head.parameters()),
            "best_auc_val": best["auc"], "best_epoch": best["epoch"],
            "state": best["state"], "history": history}


def main():
    args = _parse()
    cache = FeatureCache(args.cache_dir)
    val_cache = FeatureCache(args.val_cache_dir)
    if not cache.has_freq or not val_cache.has_freq:
        raise SystemExit("one of the caches has no frequency view -- re-render with freq.enabled")

    train = load_manifest(load_dataset_config(args.train_dataset).manifest,
                          load_dataset_config(args.train_dataset).split)
    val = load_manifest(load_dataset_config(args.val_dataset).manifest,
                        load_dataset_config(args.val_dataset).split)
    tr = _read(cache, train, args.fit_view, args.report_epoch)
    va = _read(val_cache, val, args.fit_view, args.report_epoch)

    arms = {
        "spatial_only": (tr["f_fit"], va["f_fit"]),
        "concat_freq": (torch.cat([tr["f_fit"], tr["fr_fit"]], dim=-1),
                        torch.cat([va["f_fit"], va["fr_fit"]], dim=-1)),
    }
    # Capacity-matched concat: shrink the hidden layer so its total parameter
    # count lands at the spatial-only head's, so a concat gain is signal and
    # not just the ~25% extra capacity. `ProbeHead` with n_layers=2 is
    # input -> hidden -> 1, so params = 2*D + h*(D+2) + 1.
    d0 = tr["f_fit"].shape[1]
    d1 = d0 + tr["fr_fit"].shape[1]
    h0 = args.hidden
    params0 = 2 * d0 + h0 * (d0 + 2) + 1
    h_match = round((params0 - 2 * d1 - 1) / (d1 + 2))
    arms["concat_matched"] = arms["concat_freq"]
    concat_hidden = {"concat_freq": args.hidden, "concat_matched": max(h_match, 8)}

    results = []
    for name, (ftr, fva) in arms.items():
        r = _fit(name, ftr, tr["labels"], fva, va["labels"],
                 args.epochs, concat_hidden.get(name, args.hidden), args.seed)
        # retention: selected head scored on the REPORT (degraded) view, paired.
        rep_feat = torch.cat([tr["f_rep"], tr["fr_rep"]], dim=-1) if name != "spatial_only" else tr["f_rep"]
        val_rep = torch.cat([va["f_rep"], va["fr_rep"]], dim=-1) if name != "spatial_only" else va["f_rep"]
        head = ProbeHead(ftr.shape[1], concat_hidden.get(name, args.hidden), 2, 0.0)
        head.load_state_dict(r["state"]); head.eval()
        with torch.no_grad():
            r["auc_retention_train"] = _auc(head(rep_feat).cpu(), tr["labels"].cpu())
            r["auc_retention_val"] = _auc(head(val_rep).cpu(), va["labels"].cpu())
        results.append({k: v for k, v in r.items() if k != "state"})

    summary = {"args": vars(args), "arms": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"fit view: {args.fit_view}   report (retention) view: {args.report_epoch}")
    for r in results:
        print(f"  {r['name']:12s} params={r['params']:>7}  held-out AUC={r['best_auc_val']:.5f} "
              f"@epoch {r['best_epoch']:>2}  retention AUC={r['auc_retention_val']:.5f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
