"""Inference: a directory of images -> a confidence score per image.

Output JSON:

[
  {"image_path": "/abs/path/a.png", "pred": 0.93},
  {"image_path": "/abs/path/b.jpg", "pred": 0.07}
]

pred is P(generated) in [0, 1].
"""

from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.dataset import ImageFolderDataset, collate
from eval.detectors import build_detector, resolve_device
from common.io import write_json


def predict_dir(
    image_dir: str | Path,
    detector_cfg,
    batch_size: int = 32,
    num_workers: int = 4,
) -> list[dict]:
    """Score every image under image_dir. Returns the records listed above.

    `detector_cfg` is a DetectorConfig -- the same object run_eval uses, so a
    detector is defined once and both entry points read that one definition.
    """
    detector = build_detector(detector_cfg)
    device = resolve_device(detector_cfg.device)
    # preprocess_fn(), never detector.preprocess -- see FrozenDetector.preprocess_fn.
    # `aux_fn()` is the second read a fused detector needs; None for every other.
    dataset = ImageFolderDataset(
        image_dir, preprocess=detector.preprocess_fn(), aux=detector.aux_fn()
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=collate,
    )

    records = []
    for batch, metas in tqdm(loader, desc=detector.name):
        preds = detector.score(batch.to(device)).float().cpu().numpy()
        records.extend(
            {"image_path": meta["image_path"], "pred": float(pred)}
            for meta, pred in zip(metas, preds)
        )
    return records


def write_predictions(records: list[dict], out_path: str | Path) -> None:
    write_json(records, out_path)
