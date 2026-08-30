"""Would weighting the auxiliary head more push GRACE-D past retention 1.0?

    python scripts/sweep_beta.py train/configs/train/dinov3_discrepancy.yaml --run disc_gate_ctrl

Stage 2 learns one scalar, `beta`, in a fixed additive fusion:

    logit = logit_main + beta * aux_logit

so "weight the aux head more" is a question about ONE NUMBER, and answering it
does not need another training run. This scores `logit_main`, `aux_logit` and the
clean-view logit once per held-out set, then sweeps `beta` arithmetically over
the frozen arrays. Every point on the curve is the number a stage-2 run with that
beta would have reported at validation.

Read it as a bound, not a proposal. The curve's peak is the best the fusion could
do at ANY weighting, so:

  * peak retention > 1.0 => the drift carries evidence the clean image does not,
    exactly as `grace_adapter.models.discrepancy` claims, and the training objective is
    failing to reach a weighting that exists. That is a fixable problem and this
    says what to fix it to.
  * peak at or near beta = 0 => the aux logit is redundant given the main head
    on these images. No weighting rescues it, and a forced-beta training arm
    would be spending GPU time to rediscover this curve's shape.

`auc_aux` being well above chance does NOT imply a positive peak: the aux head
can be individually predictive and still carry nothing the main head is not
already using. That distinction is the entire point of scoring the fusion rather
than the branch.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train.cache.reader import FeatureCache
from train.config import load_discrepancy_config
from grace_adapter.models.discrepancy import FusedHead
from grace_adapter.models.factory import build_discrepancy_head, load_adapter
from eval.splits import build_split
from train.loop import _cache_loader_cfg, _expect_spec, _load_val_sets, _to_float
from train.data import build_loader
from load_data.config import load_dataset_config
from eval.config import load_detector_config
from load_data.manifest import load_manifest
from eval.detectors import build_detector
from eval.metrics import retention, roc_auc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="the stage-2 config the run was trained from")
    p.add_argument("--run", required=True, help="run_id under out_dir holding discrepancy.pt")
    p.add_argument("--adapter", help="override adapter_checkpoint (defaults to the one in the checkpoint)")
    p.add_argument(
        "--betas", default="-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,2,4,8",
        help="comma-separated beta values to score",
    )
    p.add_argument("--out", help="write the sweep to this JSON path")
    return p.parse_args()


@torch.no_grad()
def _scores(fused, adapter, split, cache, manifest, epoch, cfg, device) -> dict:
    """`main`, `aux` and `clean` logits plus labels for one rendered epoch.

    The clean view is scored through the same head, because retention's
    denominator has to be this detector's own clean-image AUC on these images --
    not the number stage 1 happened to report against a different set.
    """
    loader_cfg = _cache_loader_cfg(cfg)
    main, aux, clean, labels = [], [], [], []
    for batch in build_loader(
        loader_cfg, cache, manifest, None, epoch, shuffle=False,
        with_taps=adapter.reads_taps,
    ):
        f_deg = _to_float(batch["f_deg"], device)
        sev = batch["severity"].to(device).float()
        taps = _to_float(batch["taps_deg"], device) if adapter.reads_taps else None
        delta = adapter(f_deg, severity=sev, taps=taps) - f_deg
        tap_drift = adapter.tap_drift(taps) if fused.aux.n_taps else None
        main.append(split.head(f_deg + delta).cpu().numpy())
        aux.append(fused.aux(delta, sev, tap_drift).cpu().numpy())
        clean.append(split.head(_to_float(batch["f_clean"], device)).cpu().numpy())
        labels.append(batch["label"].numpy())
    return {k: np.concatenate(v) for k, v in
            dict(main=main, aux=aux, clean=clean, labels=labels).items()}


def main():
    args = parse_args()
    cfg = load_discrepancy_config(args.config)
    betas = [float(b) for b in args.betas.split(",")]

    ckpt_path = Path(cfg.out_dir) / args.run / "discrepancy.pt"
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg.adapter_checkpoint = args.adapter or payload["adapter_checkpoint"]

    split = build_split(
        build_detector(load_detector_config(cfg.detector)), cfg.split, **cfg.split_args
    )
    device = next(split.parameters()).device
    spec = split.feature_spec
    tap_spec = split.tap_spec()
    adapter = load_adapter(cfg.adapter_checkpoint, spec, tap_spec).to(device).eval()

    from train.config import DiscrepancyConfig
    fused = FusedHead(
        build_discrepancy_head(
            spec, DiscrepancyConfig(**payload["discrepancy_cfg"]),
            payload.get("n_taps", 0),
        )
    ).to(device)
    fused.load_state_dict(payload["state_dict"])
    fused.eval()
    learned = float(fused.beta.detach())

    expect = _expect_spec(
        split, tap_spec if adapter.reads_taps else None, cfg.crop.fingerprint()
    )
    dataset_cfg = load_dataset_config(cfg.dataset)
    sets = [(
        "held_out_degradations",
        FeatureCache(cfg.cache_dir, expect=expect),
        load_manifest(dataset_cfg.manifest, dataset_cfg.split),
    )] + [
        (f"held_out_images/{name}", c, m) for name, c, m in _load_val_sets(cfg, spec, expect)
    ]

    print(f"adapter   {cfg.adapter_checkpoint}")
    print(f"stage 2   {ckpt_path}  (learned beta {learned:+.4f})\n")

    out = {"run": args.run, "adapter": cfg.adapter_checkpoint,
           "learned_beta": learned, "axes": {}}
    for name, cache, manifest in sets:
        epoch = list(cache.epochs())[-1]
        s = _scores(fused, adapter, split, cache, manifest, epoch, cfg, device)
        y = s["labels"]
        if len(np.unique(y)) < 2:
            continue
        auc_main = roc_auc(s["main"], y)
        auc_aux = roc_auc(s["aux"], y)
        auc_clean = roc_auc(s["clean"], y)
        rows = []
        for b in betas:
            auc_f = roc_auc(s["main"] + b * s["aux"], y)
            rows.append({"beta": b, "auc_fused": auc_f,
                         "retention": retention(auc_f, auc_clean)})
        best = max(rows, key=lambda r: r["auc_fused"])
        out["axes"][name] = {
            "epoch": int(epoch), "n": int(len(y)),
            "auc_main": auc_main, "auc_aux": auc_aux, "auc_clean": auc_clean,
            "retention_main": retention(auc_main, auc_clean),
            "sweep": rows, "best": best,
        }

        print(f"=== {name}  (epoch {epoch}, n={len(y)})")
        print(f"    auc_clean {auc_clean:.5f}   auc_main {auc_main:.5f} "
              f"(retention {retention(auc_main, auc_clean):.5f})   auc_aux {auc_aux:.5f}")
        print(f"    {'beta':>8} {'auc_fused':>11} {'retention':>11}")
        for r in rows:
            mark = "  <- best" if r is best else ""
            print(f"    {r['beta']:>8.3f} {r['auc_fused']:>11.5f} "
                  f"{r['retention']:>11.5f}{mark}")
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
