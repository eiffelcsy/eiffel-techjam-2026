"""Two ways to get (f_degraded, f_clean) pairs. Same interface, one config flag.

`source: cache`  -- both features read from disk. No images are decoded, no trunk
                    runs, nothing touches a GPU except the adapter. A step is two
                    memmap reads and a 2-layer MLP.
`source: live`   -- degrade and run the trunk in the loop; clean features still
                    come from the cache. Slower per step, but the degradation
                    distribution is unbounded.

`live` is the control, not dead code: it is how you check that a `cache`-trained
adapter is not exploiting the finite epoch set, and it is what runs when a
detector is too expensive to pre-render or the disk is too small. Both modes draw
from the same `EpochSchedule`, so a live run at epoch 7 sees exactly the images a
cached run at epoch 7 would -- that equality is what makes them comparable, and
it is worth a test.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from grace.cache.reader import FeatureCache
from grace.cache.schedule import EpochSchedule
from pipeline.data.dataset import load_normalized


def _collate(batch):
    """Stack every field; `index` stays an int64 tensor for cache lookups."""
    out = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        out[key] = torch.stack(values) if torch.is_tensor(values[0]) else torch.tensor(values)
    return out


class CachedPairDataset(Dataset):
    """(f_deg, f_clean, label, severity, index) for one epoch, straight off disk.

    One instance per epoch -- the epoch selects the degraded view, and epochs are
    the axis along which the corruption varies. Shuffling *within* an epoch is
    free and expected; manifest order only matters at write time.
    """

    def __init__(self, cache: FeatureCache, manifest, epoch: int):
        self.cache = cache
        self.epoch = epoch
        self.index = np.asarray(manifest.index, dtype=np.int64)
        self.labels = np.asarray(manifest["label"], dtype=np.int64)
        severity = cache.recipes(epoch)["severity"]
        self.severity = severity.loc[self.index].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        idx = self.index[i : i + 1]
        return {
            "f_deg": self.cache.degraded(idx, self.epoch)[0],
            "f_clean": self.cache.clean(idx)[0],
            "label": int(self.labels[i]),
            "severity": float(self.severity[i]),
            "index": int(self.index[i]),
        }


class LivePairDataset(Dataset):
    """(degraded image tensor, f_clean, label, severity, index).

    The trunk runs in the loop; the clean target still comes from the cache,
    because it is constant and there is no reason to recompute it.
    """

    def __init__(self, cache: FeatureCache, manifest, schedule: EpochSchedule,
                 epoch: int, preprocess):
        self.cache = cache
        self.schedule = schedule
        self.epoch = epoch
        self.preprocess = preprocess
        self.paths = manifest["path"].tolist()
        self.index = np.asarray(manifest.index, dtype=np.int64)
        self.labels = np.asarray(manifest["label"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.index[i])
        img = load_normalized(self.paths[i])
        img, recipe = self.schedule.apply(img, idx, self.epoch)
        return {
            "image": self.preprocess(img),
            "f_clean": self.cache.clean(self.index[i : i + 1])[0],
            "label": int(self.labels[i]),
            "severity": float(self.schedule.severity_of(recipe)),
            "index": idx,
        }


def build_loader(cfg, cache, manifest, schedule, epoch: int, preprocess=None,
                 shuffle: bool = True) -> DataLoader:
    """Pick the dataset by `cfg.source` and wrap it in a DataLoader.

    `cache.worker_init` is passed in both modes: memmaps are opened per worker,
    never inherited across a fork.
    """
    if cfg.source == "cache":
        dataset = CachedPairDataset(cache, manifest, epoch)
    elif cfg.source == "live":
        if preprocess is None:
            raise ValueError("source: live needs the detector's preprocess_fn()")
        dataset = LivePairDataset(cache, manifest, schedule, epoch, preprocess)
    else:
        raise ValueError(f"source must be 'cache' or 'live', got {cfg.source!r}")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        worker_init_fn=lambda w: cache.worker_init(w),
        drop_last=True,     # the sliced-Wasserstein term is a batch statistic
    )
