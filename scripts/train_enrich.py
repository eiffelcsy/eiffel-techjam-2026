"""Stage 2: train the frequency enricher against a FROZEN stage-1 adapter.

    python scripts/train_enrich.py train/configs/train/dinov3_enrich.yaml
    python scripts/train_enrich.py train/configs/train/dinov3_enrich.yaml --finetune-head --run-id dinov3_enrich_ft

The adapter stays frozen; only the enricher trains. This reads the image a
second time, in a basis the trunk's resize threw away -- which is why it can
exceed the restoration ceiling, and why it needs a cache rendered with
`freq.enabled`.

`validation.step_0` is scored before the first optimizer
step, when every expert's output projection is still zero. `auc_fused` there must
equal `auc_corrected` exactly -- that is E10, measured rather than asserted, and
if the two differ the module is not wired where it claims to be and nothing after
it is a measurement.

`--finetune-head` trains a copy of the detector's head
alongside the enricher. Run BOTH and report both. Which one is the honest
headline depends on how much of the enrichment lies in directions the frozen
head can read.
"""

import argparse

from train.config import load_enrich_config
from eval.splits import build_split
from train.loop import train_enrich
from train.tracker import add_wandb_args, apply_wandb_args
from load_data.config import load_dataset_config
from eval.config import load_detector_config
from load_data.manifest import load_manifest
from eval.detectors import build_detector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--adapter", help="override adapter_checkpoint")
    p.add_argument("--run-id")
    p.add_argument("--dataset", help="override the dataset config path")
    p.add_argument("--epochs", type=int, help="override epochs (used by --smoke)")
    p.add_argument(
        "--finetune-head", action="store_true",
        help="E14's second arm: train a copy of the detector head too",
    )
    p.add_argument(
        "--lam-anchor", type=float,
        help="override lam_anchor; 0 is E15's unanchored arm",
    )
    # The three enricher axes with a shipped experiment behind them, as flags so
    # a sweep is N invocations rather than N near-identical config files -- the
    # same reason `--seed` / `--run-id` exist on train_adapter.py. Anything else
    # about the enricher is edited in the config, where it is written down.
    p.add_argument(
        "--n-bands", type=int,
        help="E11: 1 is the single-branch control against the HF/LF pair",
    )
    p.add_argument(
        "--top-k", type=int,
        help="E13: keep the k lowest radial coefficients per channel",
    )
    p.add_argument(
        "--no-pos-emb", action="store_true",
        help="E13: drop the per-cell position embedding, making the cells a set",
    )
    add_wandb_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_enrich_config(args.config)
    if args.adapter:
        cfg.adapter_checkpoint = args.adapter
    if args.run_id:
        cfg.run_id = args.run_id
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.finetune_head:
        cfg.finetune_head = True
    if args.lam_anchor is not None:
        cfg.lam_anchor = args.lam_anchor
    if args.n_bands is not None:
        cfg.enricher.n_bands = args.n_bands
    if args.top_k is not None:
        cfg.enricher.top_k = args.top_k
    if args.no_pos_emb:
        cfg.enricher.pos_emb = False
    # Re-run the dataclass validation the overrides could have invalidated --
    # `n_heads` dividing `d_model`, `top_k >= 1`. `__post_init__` does not fire
    # on attribute assignment, and an override that skipped it would fail deep
    # inside the attention instead of at second zero.
    cfg.enricher.__post_init__()
    apply_wandb_args(cfg, args)

    dataset_cfg = load_dataset_config(args.dataset or cfg.dataset)
    manifest = load_manifest(dataset_cfg.manifest, dataset_cfg.split)
    split = build_split(
        build_detector(load_detector_config(cfg.detector)), cfg.split, **cfg.split_args
    )

    summary = train_enrich(cfg, split, manifest)
    gates = ", ".join(f"{g:.3f}" for g in summary["gates"])
    beta = summary.get("beta")
    beta_s = f"  beta={beta:.4f}" if beta is not None else ""
    print(
        f"{cfg.run_id}: gates=[{gates}]  "
        f"head={'fine-tuned' if summary['finetune_head'] else 'frozen'}{beta_s}"
    )
    for axis, rows in summary["validation"].items():
        print(f"  {axis}:")
        for name, row in rows.items():
            delta = row["auc_fused"] - row["auc_corrected"]
            aux = row.get("auc_aux")
            aux_s = f"  aux={aux:.4f}" if aux is not None else ""
            print(
                f"    {name}: +grace={row['auc_corrected']:.4f}  "
                f"+grace-freq={row['auc_fused']:.4f}  ({delta:+.4f}){aux_s}  "
                f"|enrichment|={row['enrichment_norm']:.3f}"
            )
    _assert_identity_at_step_0(summary)


def _assert_identity_at_step_0(summary: dict) -> None:
    """E10, checked here as well as in `tests/test_enricher_identity.py`.

    The test proves the module is an identity at init; this proves the TRAINING
    RUN wired it as one -- against the real adapter, the real head and the real
    cached features. A unit test cannot catch an enricher spliced in at the wrong
    point, and that is the failure worth catching before four hours of stage-2
    sweeps get read as a result.
    """
    bad = [
        f"{name}: {row['auc_corrected']:.6f} vs {row['auc_fused']:.6f}"
        for name, row in summary["validation"].get("step_0", {}).items()
        if row["auc_corrected"] != row["auc_fused"]
    ]
    if bad:
        raise SystemExit(
            "IDENTITY BROKEN AT INITIALIZATION. The enricher's output projections "
            "are zero at step 0, so `fused` must equal `f_corrected` bit for bit "
            "and these two AUCs must be identical:\n  "
            + "\n  ".join(bad)
            + "\nThe enricher is not spliced where it claims to be, and every "
            "number from this run is against a model nobody benchmarked."
        )


if __name__ == "__main__":
    main()
