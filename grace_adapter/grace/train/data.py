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
from preprocessing.dataset import load_normalized


class _WorkerInit:
    """Picklable `worker_init_fn`. Deliberately not a lambda or a bound method.

    DataLoader hands this to each worker, and on every platform whose start
    method is `spawn` -- macOS by default, and Linux from Python 3.14 -- that
    means *pickling* it. A lambda closing over `cache` is a local object and
    dies with `Can't get local object 'build_loader.<locals>.<lambda>'` on the
    first batch, after the model is loaded and the run has started.

    Same reasoning as `eval.detectors.hf._ProcessorPreprocess`: anything
    crossing into a worker is module-level and holds only what it needs.
    """

    def __init__(self, cache):
        self.cache = cache

    def __call__(self, worker_id: int) -> None:
        self.cache.worker_init(worker_id)


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

    `with_taps` adds `taps_deg`, one more memmap read. The cache also holds
    clean taps, but nothing in the objective reads them, so they are not loaded.

    `with_freq` adds `freq_deg`, the DCT view of the same degraded window, for
    stage-2 enrichment. Also not loaded by default: it is ~7x the width of a
    768-d feature row, so a stage-1 run that never reads it should not pay for
    it. Same rule as the taps -- the flag is what the objective asks for, and
    `assert_freq_available` is what refuses a cache that cannot supply it.
    """

    def __init__(self, cache: FeatureCache, manifest, epoch: int, with_taps: bool = False,
                 with_freq: bool = False):
        self.cache = cache
        self.epoch = epoch
        self.with_taps = with_taps
        self.with_freq = with_freq
        self.index = np.asarray(manifest.index, dtype=np.int64)
        self.labels = np.asarray(manifest["label"], dtype=np.int64)
        severity = cache.recipes(epoch)["severity"]
        self.severity = severity.loc[self.index].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        idx = self.index[i : i + 1]
        item = {
            "f_deg": self.cache.degraded(idx, self.epoch)[0],
            "f_clean": self.cache.clean(idx)[0],
            "label": int(self.labels[i]),
            "severity": float(self.severity[i]),
            "index": int(self.index[i]),
        }
        if self.with_taps:
            item["taps_deg"] = self.cache.taps(idx, self.epoch)[0]
        if self.with_freq:
            item["freq_deg"] = self.cache.freq(idx, self.epoch)[0]
        return item


class LivePairDataset(Dataset):
    """(degraded image tensor, f_clean, label, severity, index).

    The trunk runs in the loop; the clean target still comes from the cache,
    because it is constant and there is no reason to recompute it.
    """

    def __init__(self, cache: FeatureCache, manifest, schedule: EpochSchedule,
                 epoch: int, preprocess, with_taps: bool = False, crop=None):
        self.cache = cache
        self.schedule = schedule
        self.epoch = epoch
        self.preprocess = preprocess
        self.crop = crop
        """Must match the crop the cache's clean view was rendered under, or
        `f_clean` is the target for a different window than `image` shows."""
        self.with_taps = with_taps
        """Accepted for interface parity with `CachedPairDataset` and unused:
        the degraded taps come out of the same live `trunk_with_taps` call that
        produces `f_deg` in the loop, and nothing else reads taps here."""
        self.paths = manifest["path"].tolist()
        self.index = np.asarray(manifest.index, dtype=np.int64)
        self.labels = np.asarray(manifest["label"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.index[i])
        img = load_normalized(self.paths[i])
        img, recipe = self.schedule.apply(img, idx, self.epoch)
        if self.crop is not None:
            img = self.crop(img, idx)
        item = {
            "image": self.preprocess(img),
            "f_clean": self.cache.clean(self.index[i : i + 1])[0],
            "label": int(self.labels[i]),
            "severity": float(self.schedule.severity_of(recipe)),
            "index": idx,
        }
        return item


def build_loader(cfg, cache, manifest, schedule, epoch: int, preprocess=None,
                 shuffle: bool = True, with_taps: bool = False, crop=None,
                 with_freq: bool = False) -> DataLoader:
    """Pick the dataset by `cfg.source` and wrap it in a DataLoader.

    `cache.worker_init` is passed in both modes: memmaps are opened per worker,
    never inherited from the parent. Under `spawn` the parent's are not
    inherited at all -- `FeatureCache.__getstate__` drops them rather than
    attempting to pickle a memmap -- and `worker_init` is what re-opens them on
    the other side.
    """
    if cfg.source == "cache":
        dataset = CachedPairDataset(
            cache, manifest, epoch, with_taps=with_taps, with_freq=with_freq
        )
    elif cfg.source == "live":
        if with_freq:
            # Not an oversight to fix later: stage 2 is `source: cache` by
            # construction (`_cache_loader_cfg`), and a live frequency read would
            # have to re-derive the crop draw in the loop to stay the same window
            # the cached clean features are of.
            raise ValueError(
                "source: live cannot supply the frequency view -- render it with "
                "scripts/build_cache.py and train the enricher against the cache."
            )
        if preprocess is None:
            raise ValueError("source: live needs the detector's preprocess_fn()")
        dataset = LivePairDataset(
            cache, manifest, schedule, epoch, preprocess, with_taps=with_taps, crop=crop
        )
    else:
        raise ValueError(f"source must be 'cache' or 'live', got {cfg.source!r}")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        worker_init_fn=_WorkerInit(cache),
        drop_last=True,     # every logged step on the same batch size, so the
                            # per-term scalars are comparable along the run
    )
