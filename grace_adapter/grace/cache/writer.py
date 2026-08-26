"""Offline feature extraction: one trunk pass per (image, view), ever.

Layout on disk -- flat .npy memmaps in shards, not HDF5 or LMDB. The simplest
thing that supports random access and multi-worker reads.

    cache/{detector}/
    |-- spec.json               CacheSpec, incl. the four fingerprints
    |-- index.npy               int64, row -> manifest index (shared by all views)
    |-- clean/
    |   |-- feats_00000.npy     (rows_in_shard, *feature_shape) float16
    |   `-- .done
    |-- epoch=000/
    |   |-- feats_00000.npy
    |   |-- recipes.parquet     index, level, recipe label, transforms, severity
    |   `-- .done
    `-- epoch=001/ ...

`index.npy` is written once and shared: every view holds the same images in the
same manifest order, so row `r` means the same image in every view. That is what
makes `f_clean` and `f_deg` for one image a single lookup at the same row, and it
is the property `tests/test_cache_alignment.py` checks.

`recipes.parquet` is not bookkeeping. It carries the per-image recipe *and its
severity*, which makes retention-recovered-per-transform a groupby rather than a
re-run, and supplies the label-free severity target for free.

Never shuffle. Never rebuild the manifest afterwards. A view is finalized by a
`.done` marker written last, so an interrupted render can never be mistaken for
a complete one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from grace.cache.schedule import EpochSchedule
from grace.cache.spec import DONE_FILE, INDEX_FILE, CacheSpec, view_name
from grace.splits.base import SplitDetector
from pipeline.data.dataset import AIGCDataset, collate

RECIPE_FILE = "recipes.parquet"


class ShardWriter:
    """Append rows into fixed-size .npy shards, in order.

    Ordered append only -- no seeking, no partial rewrite. Shards are
    preallocated with `open_memmap`, so the header is valid from the first byte
    and a crashed render leaves a readable but unmarked directory.
    """

    def __init__(self, view_dir: str | Path, spec: CacheSpec):
        self.dir = Path(view_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.row = 0
        self._shard_id = -1
        self._shard = None

    def _rows_in(self, shard_id: int) -> int:
        start = shard_id * self.spec.shard_size
        return min(self.spec.shard_size, self.spec.n - start)

    def _ensure(self, shard_id: int):
        if shard_id != self._shard_id:
            self._shard = np.lib.format.open_memmap(
                self.dir / f"feats_{shard_id:05d}.npy",
                mode="w+",
                dtype=np.dtype(self.spec.feature.dtype),
                shape=(self._rows_in(shard_id), *self.spec.feature.shape),
            )
            self._shard_id = shard_id
        return self._shard

    def write(self, features: np.ndarray) -> None:
        """Append a batch. Splits across a shard boundary when it straddles one."""
        if features.shape[1:] != self.spec.feature.shape:
            raise ValueError(
                f"trunk emitted {features.shape[1:]}, spec declares "
                f"{self.spec.feature.shape}"
            )
        written = 0
        while written < len(features):
            shard_id, offset = divmod(self.row, self.spec.shard_size)
            take = min(len(features) - written, self.spec.shard_size - offset)
            shard = self._ensure(shard_id)
            shard[offset : offset + take] = features[written : written + take]
            written += take
            self.row += take

    def finalize(self) -> None:
        if self.row != self.spec.n:
            raise RuntimeError(f"wrote {self.row} rows, expected {self.spec.n}")
        if self._shard is not None:
            self._shard.flush()
            self._shard = None
        (self.dir / DONE_FILE).write_text("ok", encoding="utf-8")


def is_complete(view_dir: str | Path) -> bool:
    return (Path(view_dir) / DONE_FILE).exists()


@torch.no_grad()
def build_view(
    split: SplitDetector,
    manifest,
    view_dir: str | Path,
    spec: CacheSpec,
    schedule: EpochSchedule | None = None,
    epoch: int | None = None,
    batch_size: int = 32,
    num_workers: int = 8,
    device=None,
) -> None:
    """Render one view: the clean pass if `epoch is None`, else that epoch's.

    The only difference between the clean view and a degraded one is whether the
    dataset is handed a condition -- which is the whole point: clean features are
    not a special mechanism, they are epoch `None` of the same loop.
    """
    split.assert_frozen()
    device = device or next(split.parameters()).device
    condition = _EpochCondition(schedule, epoch) if epoch is not None else None
    dataset = AIGCDataset(manifest, preprocess=split.preprocess_fn(), condition=condition)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,                 # manifest order IS the row order. Never shuffle.
        collate_fn=collate,
    )
    writer = ShardWriter(view_dir, spec)
    records = []
    for batch, metas in tqdm(loader, desc=Path(view_dir).name, leave=False):
        f = split.trunk(batch.to(device))
        writer.write(f.to(spec.feature.torch_dtype).cpu().numpy())
        records.extend(metas)
    writer.finalize()

    if epoch is not None:
        rows = [
            {
                "index": m["index"],
                "level": condition.level_of(m["index"]),
                "recipe": m["recipe"],
                "transforms": list(m["transforms"]),
                "severity": schedule.severity_for(m["index"], epoch),
            }
            for m in records
        ]
        pd.DataFrame(rows).to_parquet(Path(view_dir) / RECIPE_FILE, index=False)


class _EpochCondition:
    """Adapts an `EpochSchedule` to the harness's per-condition Dataset interface.

    `AIGCDataset` calls `condition(img, index)` and reads `condition.id`. The
    schedule picks a different level per image, so this dispatches per call
    rather than holding one `Condition`.
    """

    def __init__(self, schedule: EpochSchedule, epoch: int):
        self.schedule = schedule
        self.epoch = epoch
        self.id = view_name(epoch)

    def level_of(self, index: int) -> int:
        return self.schedule.level_for(index, self.epoch)

    def __call__(self, img, index: int):
        return self.schedule.apply(img, index, self.epoch)


def build_cache(
    split: SplitDetector,
    manifest,
    root: str | Path,
    spec: CacheSpec,
    schedule: EpochSchedule,
    epochs,
    batch_size: int = 32,
    num_workers: int = 8,
    device=None,
) -> CacheSpec:
    """Render the clean view plus every requested epoch, resumably.

    Resumable at view granularity: a view whose `.done` marker exists is skipped.
    Rendering a dozen epochs of a large split is hours, and it will be
    interrupted.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / INDEX_FILE, np.asarray(manifest.index, dtype=np.int64))

    views = [None, *epochs]
    for epoch in views:
        view_dir = root / view_name(epoch)
        if is_complete(view_dir):
            continue
        build_view(
            split, manifest, view_dir, spec,
            schedule=schedule, epoch=epoch,
            batch_size=batch_size, num_workers=num_workers, device=device,
        )

    from dataclasses import replace

    spec = replace(spec, views=tuple(view_name(e) for e in views))
    spec.save(root)
    return spec
