"""The 32x32 round trip: does this head read traces, or content?

    python scripts/misc/roundtrip_check.py --detector eval/configs/detectors/<d>.yaml \
                                      --dataset  load_data/configs/datasets/<ds>.yaml

Downscale each image to 32x32, scale it back, and re-score. No generation trace
of any kind survives that -- the artefacts a forensic detector reads live at the
pixel scale and 32x32 is far below it -- so a head that is genuinely forensic
must fall toward chance. A head that holds its AUC is reading content, and its
retention curve will look excellent for reasons that have nothing to do with
robustness.

The finding this operationalises was made on SID_Set, before this project moved
to NTIRE and then to WildFake: under `resize`, a probe reached 0.9999 val AUC and
still held 0.985 after the round trip, while retention stayed at 100% through
JPEG-30, blur sigma=2.0 and a 4x downscale. That detector had nothing for GRACE
to repair because it was never damaged in the first place.

    survives   AUC stays high  -> CONTENT classifier. Stop.
    fails      AUC -> ~0.5      -> forensic. Proceed.

"Fails" is the pass condition, which is why this script's exit code is inverted
from the usual reading: it exits non-zero when the head SURVIVES.

Run it on a validation split, never on the reported benchmark -- it is a property
of the head, and the benchmark is not where head properties get established.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_data.config import load_dataset_config
from eval.config import load_detector_config
from preprocessing.dataset import AIGCDataset, collate
from load_data.manifest import load_manifest, sample_eval_subset
from eval.detectors import build_detector
from eval.metrics import roc_auc

SURVIVES_ABOVE = 0.70
"""AUC above which the head is judged to have survived the round trip.

Chance is 0.5 and a forensic head should land near it. 0.70 is deliberately
generous -- the question is "did this collapse", not "did this collapse to
exactly chance" -- and it is far below the 0.985 that condemned the whole-image
protocol. A head landing between 0.5 and 0.7 is reported as marginal rather than
waved through.
"""


class RoundTrip:
    """Picklable `(image, index) -> image`, in the dataset's `crop` slot.

    Sits where the multi-scale window would, i.e. after any degradation and
    before preprocessing, so what the detector receives differs from a normal
    run in exactly one way.
    """

    def __init__(self, size: int = 32):
        self.size = int(size)

    def __call__(self, img: Image.Image, index: int) -> Image.Image:
        w, h = img.size
        small = img.resize((self.size, self.size), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


@torch.no_grad()
def score(detector, manifest, crop, batch_size: int, num_workers: int, device) -> tuple:
    loader = DataLoader(
        AIGCDataset(manifest, preprocess=detector.preprocess_fn(), crop=crop),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate,
    )
    scores, labels = [], []
    for batch, metas in tqdm(loader, leave=False):
        scores.append(detector.score(batch.to(device)).float().cpu().numpy())
        labels.extend(int(m["label"]) for m in metas)
    return np.concatenate(scores), np.array(labels, dtype=int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="rows per class")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    det_cfg = load_detector_config(args.detector)
    ds_cfg = load_dataset_config(args.dataset)
    detector = build_detector(det_cfg)
    device = next(detector.parameters()).device

    manifest = sample_eval_subset(
        load_manifest(ds_cfg.manifest, ds_cfg.split), args.limit, seed=0
    )
    print(f"{det_cfg.name} on {ds_cfg.name}: {len(manifest)} rows\n")

    clean_s, y = score(detector, manifest, None, args.batch_size, args.num_workers, device)
    rt_s, _ = score(
        detector, manifest, RoundTrip(args.size), args.batch_size, args.num_workers, device
    )

    clean_auc, rt_auc = roc_auc(clean_s, y), roc_auc(rt_s, y)
    kept = (rt_auc - 0.5) / (clean_auc - 0.5) if clean_auc > 0.5 else float("nan")
    survives = rt_auc > SURVIVES_ABOVE

    print(f"  clean AUC          {clean_auc:.4f}")
    print(f"  {args.size}x{args.size} round trip  {rt_auc:.4f}")
    print(f"  skill retained     {kept:.1%}")
    print()
    if survives:
        print(f"  SURVIVES (> {SURVIVES_ABOVE}). This head reads CONTENT, not generation")
        print( "  traces: no trace survives a 32x32 round trip, so whatever it is")
        print( "  scoring on is still there. Its retention curve will look excellent")
        print( "  for the wrong reason, and there is nothing here for GRACE to repair.")
    elif rt_auc > 0.6:
        print(f"  MARGINAL. Collapsed, but not to chance -- some content signal remains.")
        print( "  Usable, but report this number next to any retention claim.")
    else:
        print( "  COLLAPSES toward chance. The head is reading something that does not")
        print( "  survive resampling, which is the precondition for the rest of this")
        print( "  project meaning anything.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "detector": det_cfg.name, "dataset": ds_cfg.name, "n": int(len(manifest)),
            "size": args.size, "clean_auc": clean_auc, "roundtrip_auc": rt_auc,
            "skill_retained": kept, "survives": bool(survives),
        }, indent=2))
        print(f"\nwrote {args.out}")

    raise SystemExit(1 if survives else 0)


if __name__ == "__main__":
    main()
