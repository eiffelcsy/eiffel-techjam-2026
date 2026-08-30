"""Score a directory of images.

    python scripts/predict.py --image-dir path/to/images \
        --detector eval/configs/detectors/<detector>.yaml --out preds.json

Output: JSON list of {"image_path": str, "pred": float}, pred = P(AI-generated).
"""

import argparse

from eval.config import load_detector_config
from eval.inference import predict_dir, write_predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out", default="preds.json")
    p.add_argument("--detector", required=True, help="detector spec yaml")
    p.add_argument("--device", default=None, help="override the spec's device")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    detector_cfg = load_detector_config(args.detector)
    if args.device:
        detector_cfg.device = args.device

    records = predict_dir(
        args.image_dir, detector_cfg,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    write_predictions(records, args.out)
    print(f"wrote {args.out}: {len(records)} predictions")


if __name__ == "__main__":
    main()
