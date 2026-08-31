"""E0-freq: does the DCT spectrum separate real from generated, ABOVE the floor?

    python scripts/misc/analyze_freq.py --dataset load_data/configs/datasets/wildfake_train_val.yaml

DECISION 0. If band energies do not separate the classes by more than spectral
rolloff alone does, the frequency branch's mechanism has failed and no amount of
cross-attention rescues it -- re-scope to a simpler band-energy input, or
abandon, BEFORE a byte of the frequency cache is rendered. This script exits
non-zero when that happens, so it can be a hard stop in a driver script.

WHY A FLOOR AND NOT A THRESHOLD. WildFake's reals are COCO images that the
dataset packager downsampled to 200x200; the fakes are untouched native DALL-E 3
pixels. A downsample leaves a signature -- the spectrum rolls off at the old
Nyquist and stays near zero above it -- and that signature is *in the pixels*, so
cropping does not remove it and a frequency branch reads it perfectly. A band
result that merely beats chance is therefore not evidence about generation
traces; it is evidence about resampling history. E-rolloff measures exactly that
channel with two numbers, and everything the bands do above it is what is left.

Three things come out, and the third is the one to read:

    E-rolloff    a classifier on effective bandwidth and 95% rolloff alone
    E-freqonly   a classifier on all band energies, no spatial branch
    signatures   per-degradation band deltas -- blur/resize should DESTROY high
                 frequencies, gaussian noise should ADD them, JPEG should move
                 energy onto block-aligned coefficients. If the bands are reading
                 real spectral structure these directions come out right; if the
                 three look alike, the "bands" are one number wearing a hat.

RUNS ON THE VALIDATION SPLIT, never on the reported benchmark. Band count and
patch size are choices, and choosing them against a test-set number is selection
on the test set -- the repo rule, and the reason this defaults to
`wildfake_train_val`.

It consumes the SHIPPED extractor (`freq_branch.dct`), not a reimplementation,
so a mechanism that passes here is the mechanism the model gets.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from train.config import load_probe_config
from load_data.config import load_dataset_config
from preprocessing.dataset import load_normalized
from load_data.manifest import load_manifest, sample_eval_subset
from preprocessing.degrade.conditions import build_conditions, load_grid
from preprocessing.degrade.crop import SAMPLE_EPOCH, multiscale_crop
from eval.metrics import roc_auc
from freq_branch.dct import DEFAULT_GRID, DEFAULT_PATCH, band_masks, extract_freq

DEFAULT_BANDS = 8
"""Bands for the ANALYSIS, not for the model.

The enricher ships two experts (HF and LF) because two is what a gate can
usefully specialise over. Eight here is a resolution choice about the report: a
two-band summary cannot show *where* along the spectrum a degradation moves
energy, and "where" is the whole content of the signature table. The band-energy
classifier is run over all eight, which makes E-freqonly a generous reading of
what the spectrum carries -- deliberately, since it is being used to decide
whether to STOP.
"""


def energies(coeffs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """`(cells, n_coeffs)` and `(n_bands, n_coeffs)` -> mean energy per band.

    Mean over cells first, then per band: the spatial average is what the
    enricher's queries see pooled anyway, and a per-cell classifier would be
    measuring composition, which is the thing the crop exists to remove.
    """
    per_cell = coeffs @ masks.T                      # (cells, n_bands)
    return per_cell.mean(axis=0) / np.maximum(masks.sum(axis=1), 1.0)


def rolloff(coeffs: np.ndarray, patch: int, channels: int) -> np.ndarray:
    """Effective bandwidth and 95% rolloff, averaged over cells and channels.

    Both are read off the radially-ordered coefficient axis, where position IS
    frequency, so no second transform is needed. `bandwidth` is the
    energy-weighted mean radial position -- where the spectrum's mass sits --
    and `rolloff95` is where 95% of it is behind, which is the sharper read of
    "this was downsampled and there is nothing above here".
    """
    per = patch * patch
    blocks = coeffs.reshape(coeffs.shape[0], channels, per).mean(axis=0)  # (C, per)
    pos = np.arange(per, dtype=np.float64)
    total = blocks.sum(axis=1, keepdims=True)
    weights = blocks / np.maximum(total, 1e-12)
    bandwidth = float((weights * pos).sum(axis=1).mean())
    cumulative = np.cumsum(weights, axis=1)
    r95 = float(np.mean([np.searchsorted(row, 0.95) for row in cumulative]))
    return np.array([bandwidth, r95], dtype=np.float64)


class _Reader:
    """One image -> its band energies and rolloff under every condition.

    The crop is drawn once per image, at `SAMPLE_EPOCH`, and reused across every
    condition. That is not a shortcut: it is the same rule the render follows,
    and it is what makes the per-degradation deltas below a comparison of one
    window against itself rather than of two different pictures.
    """

    def __init__(self, conditions, crop, masks, patch, grid, channels):
        self.conditions = conditions
        self.crop = crop
        self.masks = masks
        self.patch, self.grid, self.channels = patch, grid, channels

    def __call__(self, row):
        path, index, label = row
        try:
            image = load_normalized(path)
        except Exception:
            return None
        out = {}
        for condition in self.conditions:
            view = image if condition.level == 0 else condition(image, index)[0]
            window, _ = multiscale_crop(
                view, index, SAMPLE_EPOCH, self.crop.seed,
                self.crop.s_min, self.crop.s_max, self.crop.policy,
            )
            coeffs = extract_freq(
                np.asarray(window, dtype=np.uint8), self.patch, self.grid
            )
            out[condition.id] = (
                energies(coeffs, self.masks),
                rolloff(coeffs, self.patch, self.channels),
            )
        return label, out


def _auc_of(x: np.ndarray, y: np.ndarray) -> float:
    """Cross-validated AUC of a logistic regression on `x`.

    Cross-validated because these are being compared against each other and one
    of them has 8 features while the other has 2 -- an in-sample fit would hand
    the wider one an advantage that is entirely capacity.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pred = cross_val_predict(model, x, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc(pred, y)


def separation(auc: float) -> float:
    """Distance from chance. A feature that predicts BACKWARDS separates the
    classes exactly as well as one that predicts forwards, and on this corpus the
    reals are the less-compressed half, so several of these run negative."""
    return abs(auc - 0.5) * 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="load_data/configs/datasets/wildfake_train_val.yaml",
        help="NEVER the reported benchmark -- see the module docstring",
    )
    p.add_argument(
        "--crop-from", default="train/configs/probe/dinov3_wildfake_multiscale.yaml",
        help="probe config supplying the audited crop range",
    )
    p.add_argument("--limit", type=int, default=1000, help="images per class")
    p.add_argument("--bands", type=int, default=DEFAULT_BANDS)
    p.add_argument("--patch", type=int, default=DEFAULT_PATCH)
    p.add_argument("--grid", type=int, default=DEFAULT_GRID)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--margin", type=float, default=0.05,
        help="how far E-freqonly must clear E-rolloff to count as a mechanism",
    )
    p.add_argument("--out", default="results/freq_analysis__wildfake-train-val.json")
    return p.parse_args()


def main():
    args = parse_args()
    # The crop range comes from the probe config, which is where the audit wrote
    # it. Restating it on this command line would let the falsification test
    # certify a protocol nobody trains under.
    crop = load_probe_config(args.crop_from).crop
    if not crop.enabled:
        raise SystemExit(
            f"{args.crop_from} has crop.enabled false. E0-freq is a statement "
            f"about multi-scale crops; on whole images it would measure the "
            f"dimension shortcut instead."
        )

    dataset_cfg = load_dataset_config(args.dataset)
    manifest = sample_eval_subset(
        load_manifest(dataset_cfg.manifest, dataset_cfg.split), args.limit, seed=0
    )
    grid = load_grid("preprocessing/configs/degradations.yaml")
    # Clean plus the 19 single-transform grid points. No composed levels: the
    # signature table asks what ONE transform does to the spectrum, and a mixture
    # of two answers no question at all.
    conditions = build_conditions(grid, [0, 1], seed=0)

    masks = band_masks(args.patch, args.bands, args.channels)
    reader = _Reader(conditions, crop, masks, args.patch, args.grid, args.channels)
    rows = list(zip(manifest["path"], manifest.index, manifest["label"]))

    print(
        f"{dataset_cfg.name}: {len(rows)} images x {len(conditions)} conditions, "
        f"crop {crop.s_min}-{crop.s_max}px {crop.policy}, {args.bands} bands"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [
            r for r in tqdm(
                pool.map(reader, rows), total=len(rows), desc="dct", leave=False
            ) if r is not None
        ]
    if not results:
        raise SystemExit("no images could be read -- rebuild the manifest")

    labels = np.array([label for label, _ in results], dtype=int)
    if len(np.unique(labels)) < 2:
        raise SystemExit("one class only; nothing to separate")

    band_clean = np.stack([r["clean"][0] for _, r in results])
    roll_clean = np.stack([r["clean"][1] for _, r in results])

    report = {
        "dataset": dataset_cfg.name,
        "n_images": len(results),
        "crop": {"s_min": crop.s_min, "s_max": crop.s_max, "policy": crop.policy},
        "extraction": {
            "patch": args.patch, "grid": args.grid, "bands": args.bands,
            "channels": args.channels,
        },
    }

    # -- the floor, and what stands above it ---------------------------------
    roll_auc = _auc_of(roll_clean, labels)
    freq_auc = _auc_of(band_clean, labels)
    report["e_rolloff"] = {
        "auc": roll_auc,
        "separation": separation(roll_auc),
        "bandwidth_auc": roc_auc(roll_clean[:, 0], labels),
        "rolloff95_auc": roc_auc(roll_clean[:, 1], labels),
    }
    report["e_freqonly"] = {"auc": freq_auc, "separation": separation(freq_auc)}
    report["per_band"] = {
        f"band_{b}": {
            "auc": roc_auc(band_clean[:, b], labels),
            "mean_real": float(band_clean[labels == 0, b].mean()),
            "mean_fake": float(band_clean[labels == 1, b].mean()),
        }
        for b in range(args.bands)
    }

    print("\n  BANDS, CLEAN CROPS -- each alone")
    print(f"  {'band':<10}{'AUC':>10}{'separation':>14}{'real':>12}{'fake':>12}")
    for name, row in report["per_band"].items():
        print(
            f"  {name:<10}{row['auc']:>10.4f}{separation(row['auc']):>14.4f}"
            f"{row['mean_real']:>12.4f}{row['mean_fake']:>12.4f}"
        )
    print(
        f"\n  {'E-rolloff (floor)':<26}{roll_auc:>10.4f}"
        f"{separation(roll_auc):>14.4f}"
    )
    print(f"  {'E-freqonly (all bands)':<26}{freq_auc:>10.4f}{separation(freq_auc):>14.4f}")

    # -- do the degradations move the spectrum the way physics says? ----------
    by_transform: dict[str, list] = defaultdict(list)
    for condition in conditions:
        if condition.level != 1:
            continue
        by_transform[condition.steps[0].transform].append(condition.id)

    clean_mean = band_clean.mean(axis=0)
    signatures = {}
    for transform, ids in sorted(by_transform.items()):
        stacked = np.stack(
            [r[cid][0] for _, r in results for cid in ids]
        ).reshape(len(results), len(ids), args.bands).mean(axis=(0, 1))
        delta = stacked - clean_mean
        signatures[transform] = {
            "delta_by_band": [float(d) for d in delta],
            # One number per transform for the table: the top quarter of the
            # spectrum against the bottom quarter. Blur and downscale should be
            # strongly negative here, gaussian noise strongly positive.
            "hf_delta": float(delta[-max(args.bands // 4, 1):].mean()),
            "lf_delta": float(delta[: max(args.bands // 4, 1)].mean()),
        }
    report["signatures"] = signatures

    print("\n  DEGRADATION SIGNATURES -- band energy relative to the clean crop")
    print(f"  {'transform':<20}{'HF delta':>12}{'LF delta':>12}")
    for transform, row in signatures.items():
        print(f"  {transform:<20}{row['hf_delta']:>12.4f}{row['lf_delta']:>12.4f}")

    # `distinct` is the second half of the mechanism claim. Bands that separate
    # the classes but respond identically to blur, noise and JPEG are not reading
    # the spectrum -- they are reading one scalar, and two band experts have
    # nothing to specialise over.
    hf = np.array([row["hf_delta"] for row in signatures.values()])
    report["signature_spread"] = float(hf.std())

    gap = report["e_freqonly"]["separation"] - report["e_rolloff"]["separation"]
    report["margin_over_rolloff"] = gap
    report["required_margin"] = args.margin
    passed = gap >= args.margin
    report["decision_0"] = "PROCEED" if passed else "STOP"
    print(
        f"\n  margin over the rolloff floor: {gap:+.4f} "
        f"(required {args.margin:+.4f})  ->  {report['decision_0']}"
    )
    print(f"  signature spread across transforms: {report['signature_spread']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    if not passed:
        raise SystemExit(
            "\nDECISION 0: STOP. The band energies do not separate the classes by "
            "more than spectral rolloff alone does, so whatever the spectrum "
            "carries here is resampling history rather than generation traces. "
            "The enricher would be attending over that channel. Re-scope to a "
            "simpler band-energy input, or drop the frequency branch -- "
            "before rendering the cache, which is what this gate is for."
        )


if __name__ == "__main__":
    main()
