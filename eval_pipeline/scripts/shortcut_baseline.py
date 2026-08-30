"""E-shortcut: what a classifier gets from the FILE, without looking at pixels.

    python scripts/shortcut_baseline.py --dataset configs/datasets/<ds>.yaml

Fits a logistic regression on four numbers per image -- width, height, byte size
and bits-per-pixel -- and reports its AUC. Nothing here decodes an image; this is
what the container alone gives away.

It exists because on the raw benchmark it gives away everything. Every real in
wildfake_test is exactly 200x200 (WildFake ships the COCO half pre-resized) and
the fakes are native 1024px DALL-E 3, so `max(w, h)` scores AUC 1.0000 over all
13,841 rows. Any detector reported on that set without this number beside it is
reporting an unknown mixture of forensics and file metadata.

WHAT THE ARMS DO AND DO NOT FIX. The features split in two, and conflating them
overstates the repair:

  dies under the arms   width, height, longest_side. Both arms hand every image
                        identical dimensions, so these carry literally zero
                        information about what the DETECTOR sees. Structural, not
                        normalised.

  survives into pixels  bits-per-pixel, and byte size through it. Compression
                        history is baked into the pixels as JPEG artefacts, and
                        no choice of window removes it. On wildfake_test the
                        reals sit at 2.33 bpp against the fakes' 1.34 -- so this
                        channel is live, and it is the one a frequency branch is
                        most likely to read by accident.

So the combined AUC here is a property of the DATASET, not a bound on the
detector: the detector never sees a file. Read the dimension rows as "what the
arms had to remove" and the bits-per-pixel row as "what is still there".

Run it on whatever set a number is about to be reported on, every time.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pipeline.config import load_dataset_config
from pipeline.data.manifest import load_manifest
from pipeline.eval.metrics import roc_auc

FEATURES = ("width", "height", "bytes", "bits_per_pixel")

SCALE_BOUND = ("width", "height", "longest_side", "bytes")
"""Features that die under either evaluation arm.

`bytes` belongs here rather than with compression, which is the easy mistake:
a 1024x1024 fake is 176 KB and a 200x200 real is 11 KB, and almost all of that
gap is pixel COUNT, not encoder quality. Grouping it as a compression signal
would report a surviving floor of ~1.0 when the honestly surviving channel is
bits-per-pixel at about 0.62.
"""

PIXEL_BOUND = ("bits_per_pixel",)
"""Size-normalised, so it is a statement about the encoder rather than the frame.
Baked into the pixels as JPEG artefacts; no window choice removes it."""


def probe(path: str):
    """Header read plus a stat call. No decode."""
    try:
        with Image.open(path) as im:
            w, h = im.size
        n = os.path.getsize(path)
        return [float(w), float(h), float(n), n * 8.0 / (w * h)]
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_dataset_config(args.dataset)
    df = load_manifest(cfg.manifest, split=args.split or cfg.split)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(probe, df["path"].tolist()))

    bad = [p for p, r in zip(df["path"], rows) if r is None]
    if bad:
        raise SystemExit(f"{len(bad)} unreadable, first {bad[0]!r}. Rebuild the manifest.")

    x = np.asarray(rows, dtype=np.float64)
    y = df["label"].to_numpy()
    print(f"{cfg.name}: {len(df)} rows\n")

    report = {"dataset": cfg.name, "rows": int(len(df)), "single": {}}

    # Each feature alone, so a single degenerate column is visible rather than
    # averaged into a combined score. `sep` is the distance from chance, since a
    # feature that predicts backwards (the reals here are the LESS compressed
    # half) separates the classes exactly as well as one that predicts forwards.
    def show(name: str, values: np.ndarray) -> float:
        auc = roc_auc(values, y)
        report["single"][name] = auc
        print(f"  {name:<18}{auc:>10.4f}{abs(auc - 0.5) * 2:>14.4f}")
        return auc

    column = {name: x[:, i] for i, name in enumerate(FEATURES)}
    column["longest_side"] = x[:, :2].max(axis=1)

    header = f"  {'feature':<18}{'AUC':>10}{'separation':>14}"
    print("  DIES UNDER THE ARMS -- identical dimensions there, so zero information")
    print(header)
    for name in SCALE_BOUND:
        show(name, column[name])

    print("\n  SURVIVES INTO THE PIXELS -- encoder quality, as JPEG artefacts")
    print(header)
    for name in PIXEL_BOUND:
        show(name, column[name])

    # All four together, cross-validated so the number is not the fit's own.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pred = cross_val_predict(model, x, y, cv=5, method="predict_proba")[:, 1]
    combined = roc_auc(pred, y)
    report["combined_auc"] = combined
    print(f"\n  {'combined (5-fold)':<18}{combined:>8.4f}")

    # The number that still matters after the arms are in place.
    live = max(abs(report["single"][n] - 0.5) * 2 for n in PIXEL_BOUND)
    report["surviving_separation"] = live
    verdict = (
        "the container gives nothing away" if combined < 0.55
        else f"LEAKS. Dimensions are removed by the arms; compression is NOT, and "
             f"separates at {live:.3f}. Report frequency results against that."
    )
    report["verdict"] = verdict
    print(f"\n  {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
