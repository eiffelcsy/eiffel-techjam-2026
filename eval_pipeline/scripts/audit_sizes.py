"""Native image sizes per class, and the crop range that carries no label.

    python scripts/audit_sizes.py --config configs/datasets/<name>.yaml

Why this is a gate rather than a diagnostic. Multi-scale cropping exists to
remove the resolution shortcut: on `wildfake_test`, every real is exactly
200x200 and every fake is 1024px or larger, so `max(w, h) > 512` separates the
classes at TPR 0.9994 / FPR 0.0000 and a model shown whole images reads that
instead of the content.

Cropping removes it only if the crop *size* does not reintroduce it. A window
larger than a source cannot be taken, so `multiscale_crop` clamps the draw down
to the source's short side and records the shortfall. If one class clamps more
than the other -- because its sources are systematically smaller -- then the
realized crop size is itself a classifier, and the shortcut has simply moved
from the image to the augmentation.

So the range is not a hyperparameter to pick from the plan; it is read off the
corpus. This script reports, for each candidate `s_max`, the realized-size AUC
(E-cropsize) and recommends the largest range that stays at chance.

WHICH CORPUS. Run this on the corpus whose protocol you are setting, and only
that one:

    wildfake_train  -> sets the TRAINING crop range (s_min, s_max)
    wildfake_test   -> characterises the BENCHMARK, and fixes arm (a)'s size

They are different populations and neither answer transfers. The test set's
reals are WildFake's pre-resized 200x200 COCO images; the training set's are
laion5b, imagenet, ffhq and celebahq at their own native sizes. Reading a
training range off the test audit would be setting a training hyperparameter
from the reported benchmark, which is exactly the thing not to do.

What this script does NOT do is score a detector. It reads image headers and
simulates the draw arithmetically, so every number here is a property of the
data's construction rather than of any model's performance -- a descriptive
statistic of the corpus, of the kind that belongs in the dataset section of a
writeup. Arm (a)'s size is chosen by the rule "the largest window every image
supplies from its own pixels", which is statable in advance; on this benchmark
the data makes that 200, because every real is exactly 200x200.

Sizes come from image headers, not decoded pixels, so a 100k corpus is a few
minutes of I/O rather than an hour of JPEG decode.
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.config import load_dataset_config
from pipeline.data.manifest import load_manifest
from pipeline.degrade.crop import draw_size
from pipeline.eval.metrics import roc_auc
from common.seeding import stable_seed

CANDIDATES = (128, 160, 200, 224, 256, 320, 384, 448, 512)
CHANCE = 0.55
"""E-cropsize above this counts as the label leaking through crop size.

0.55 rather than 0.5 because the estimate is finite-sample: a 20k-row audit has
a standard error of roughly 0.005 on AUC, so a threshold at 0.5 would reject
every range on noise alone. Well below the 0.9997 the raw dimensions score.
"""


def read_size(path: str) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def collect(paths: list[str], workers: int) -> list[tuple[int, int] | None]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(read_size, paths))


def percentiles(values: np.ndarray) -> dict:
    if not len(values):
        return {}
    q = np.percentile(values, [0, 1, 5, 25, 50, 75, 100])
    return {
        "n": int(len(values)),
        "min": int(q[0]), "p1": int(q[1]), "p5": int(q[2]),
        "median": int(q[4]), "p75": int(q[5]), "max": int(q[6]),
    }


def realized_sizes(
    short: np.ndarray, index: np.ndarray, s_min: int, s_max: int,
    seed: int, policy: str, epochs: int,
) -> np.ndarray:
    """Mean realized crop side per image, over `epochs` draws.

    Mirrors `multiscale_crop`'s draw exactly -- same `stable_seed` tag, same
    policy, same clamp to the short side -- without touching the pixels. If this
    ever diverges from `crop.py` the audit is measuring a protocol nobody runs.
    """
    out = np.empty((len(short), epochs), dtype=np.float64)
    for e in range(epochs):
        for i, (s, idx) in enumerate(zip(short, index)):
            rng = np.random.default_rng(stable_seed(int(idx), e, seed, "crop"))
            out[i, e] = min(draw_size(rng, s_min, s_max, policy), int(s))
    return out.mean(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="dataset spec yaml")
    p.add_argument("--split", default=None, help="override the spec's split")
    p.add_argument("--limit", type=int, default=None, help="rows per class, for a quick look")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--s-min", type=int, default=128)
    p.add_argument("--policy", default="uniform")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=4, help="draws averaged per image")
    p.add_argument("--out", default=None, help="write the report as JSON")
    p.add_argument("--max-unreadable", type=int, default=0,
                   help="tolerate this many unreadable images (they are "
                        "excluded from the audit, and reported)")
    args = p.parse_args()

    cfg = load_dataset_config(args.config)
    df = load_manifest(cfg.manifest, split=args.split or cfg.split)
    if args.limit:
        df = df.groupby("label", sort=True, group_keys=False).head(args.limit)
    print(f"{cfg.name}: {len(df)} rows")

    sizes = collect(df["path"].tolist(), args.workers)
    missing = [p for p, s in zip(df["path"], sizes) if s is None]
    if len(missing) > args.max_unreadable:
        # EXIT 2, not 1. A caller has to be able to tell "this corpus cannot
        # supply a safe crop range" (a finding about the data's shape) from
        # "some files are broken" (a problem with the download). Both used to
        # exit 1, and after_fetch.sh duly reported a corrupt PNG as proof that
        # the crop range leaked -- a confident, specific, wrong diagnosis.
        print(file=sys.stderr)
        print(
            f"{len(missing)} of {len(df)} images could not be read, the "
            f"first being {missing[0]!r}.",
            file=sys.stderr,
        )
        print(
            f"Re-extract them, or pass --max-unreadable {len(missing)} to "
            f"audit the rest and decide what to do about them separately.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if missing:
        print(f"  WARNING: skipping {len(missing)} unreadable image(s), "
              f"first {missing[0]!r}")
        keep = [s is not None for s in sizes]
        df, sizes = df[keep], [s for s in sizes if s is not None]

    wh = np.array(sizes, dtype=np.int64)
    short = wh.min(axis=1)
    longest = wh.max(axis=1)
    labels = df["label"].to_numpy()
    index = df.index.to_numpy()

    report: dict = {"dataset": cfg.name, "rows": int(len(df))}

    # --- what the raw dimensions already give away --------------------------
    report["dimension_shortcut_auc"] = roc_auc(longest.astype(float), labels)
    print(f"\ndimension shortcut (longest side as the score): "
          f"AUC {report['dimension_shortcut_auc']:.4f}")

    # --- per class and per corpus -------------------------------------------
    report["short_side"] = {
        "real": percentiles(short[labels == 0]),
        "fake": percentiles(short[labels == 1]),
    }
    print("\nshort side by class")
    for name, stats in report["short_side"].items():
        if stats:
            print(f"  {name:5s} n={stats['n']:7d}  min={stats['min']:5d}  "
                  f"p1={stats['p1']:5d}  p5={stats['p5']:5d}  "
                  f"median={stats['median']:5d}  max={stats['max']:5d}")

    by_gen: dict[str, list] = defaultdict(list)
    for g, s in zip(df["generator"], short):
        by_gen[str(g)].append(s)
    report["short_side_by_generator"] = {
        g: percentiles(np.array(v)) for g, v in sorted(by_gen.items())
    }
    print("\nshort side by generator")
    for g, stats in report["short_side_by_generator"].items():
        print(f"  {g[:28]:28s} n={stats['n']:7d}  min={stats['min']:5d}  "
              f"p5={stats['p5']:5d}  median={stats['median']:5d}")

    # --- the gate ------------------------------------------------------------
    print(f"\ncrop range audit (s_min={args.s_min}, policy={args.policy}, "
          f"{args.epochs} draws/image)")
    print(f"  {'s_max':>6} {'clamp real':>11} {'clamp fake':>11} "
          f"{'E-cropsize':>11}  verdict")

    rows = []
    for s_max in CANDIDATES:
        if s_max < args.s_min:
            continue
        realized = realized_sizes(
            short, index, args.s_min, s_max, args.seed, args.policy, args.epochs
        )
        clamp_real = float((short[labels == 0] < s_max).mean())
        clamp_fake = float((short[labels == 1] < s_max).mean())
        auc = roc_auc(realized, labels)
        safe = bool(abs(auc - 0.5) <= (CHANCE - 0.5))
        rows.append({
            "s_max": s_max, "clamp_rate_real": clamp_real,
            "clamp_rate_fake": clamp_fake, "cropsize_auc": auc, "safe": safe,
        })
        print(f"  {s_max:>6} {clamp_real:>11.3f} {clamp_fake:>11.3f} "
              f"{auc:>11.4f}  {'ok' if safe else 'LEAKS'}")

    report["candidates"] = rows
    safe = [r["s_max"] for r in rows if r["safe"]]
    report["recommended_s_max"] = max(safe) if safe else None

    if safe:
        print(f"\nrecommended: s_max = {report['recommended_s_max']} "
              f"(largest range whose realized crop size stays at chance)")
    else:
        print(f"\nNO SAFE RANGE. Every candidate leaks the label through crop "
              f"size. The corpus cannot supply a class-independent crop "
              f"distribution at s_min={args.s_min}; lower it, or fix the data.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")

    raise SystemExit(0 if safe else 1)


if __name__ == "__main__":
    main()
