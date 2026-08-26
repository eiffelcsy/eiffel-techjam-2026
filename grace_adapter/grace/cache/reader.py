"""Reading features back: memory-mapped, per-worker, aligned by manifest index.

Two rules, both easy to get wrong and both silent when you do:

  * Open memmaps lazily *inside* each DataLoader worker, never in the parent.
    Memmaps do not survive fork cleanly on every platform.
  * Index by manifest index, never by row position. The manifest index is the
    stable image identity that seeds every degradation and survives subsetting;
    row position does not. `index.npy` is the translation, and it is the same
    translation for every view.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from grace.cache.spec import CLEAN_VIEW, INDEX_FILE, CacheSpec, view_name
from grace.cache.writer import RECIPE_FILE, is_complete


class FeatureCache:
    """Random access into one detector's cache directory.

    Verifies the spec fingerprints against what the caller expects *at
    construction*, before a single feature is read.
    """

    def __init__(self, root: str | Path, expect: CacheSpec | None = None):
        self.root = Path(root)
        self._spec = CacheSpec.load(self.root)
        if expect is not None:
            self._spec.assert_compatible(expect)

        self._index = np.load(self.root / INDEX_FILE)
        # searchsorted against a sorted copy: manifest order is ascending after
        # `sample_eval_subset`'s sort_index, but nothing guarantees it forever.
        self._order = np.argsort(self._index, kind="stable")
        self._sorted = self._index[self._order]
        self._shards: dict[str, list] = {}

    @property
    def spec(self) -> CacheSpec:
        return self._spec

    @property
    def index(self) -> np.ndarray:
        """Manifest indices, in row order."""
        return self._index

    def epochs(self) -> tuple[int, ...]:
        """Which epochs were actually rendered and finalized.

        The training loop cycles over these rather than over `range(cfg.epochs)`,
        so a partially rendered cache trains on what exists instead of failing at
        epoch 9 of 12.
        """
        found = []
        for path in sorted(self.root.glob("epoch=*")):
            if is_complete(path):
                found.append(int(path.name.split("=")[1]))
        return tuple(found)

    def rows_for(self, indices) -> np.ndarray:
        """Manifest indices -> row positions. Raises on anything not cached."""
        indices = np.asarray(indices, dtype=np.int64)
        pos = np.searchsorted(self._sorted, indices)
        pos = np.clip(pos, 0, len(self._sorted) - 1)
        rows = self._order[pos]
        missing = self._index[rows] != indices
        if missing.any():
            raise KeyError(
                f"{int(missing.sum())} manifest index/indices are not in this cache, "
                f"first is {int(indices[missing][0])}. The manifest and the cache "
                f"disagree about which images exist."
            )
        return rows

    def _view(self, name: str) -> list:
        """Per-process lazy open. Called from workers, never inherited."""
        if name not in self._shards:
            view_dir = self.root / name
            if not is_complete(view_dir):
                raise FileNotFoundError(f"view {name!r} was never finished rendering")
            self._shards[name] = [
                np.load(p, mmap_mode="r") for p in sorted(view_dir.glob("feats_*.npy"))
            ]
        return self._shards[name]

    def _gather(self, name: str, rows: np.ndarray) -> torch.Tensor:
        shards = self._view(name)
        size = self._spec.shard_size
        out = np.empty((len(rows), *self._spec.feature.shape), dtype=self._spec.feature.dtype)
        shard_ids, offsets = np.divmod(rows, size)
        for shard_id in np.unique(shard_ids):
            sel = shard_ids == shard_id
            out[sel] = shards[shard_id][offsets[sel]]
        return torch.from_numpy(out)

    def clean(self, indices) -> torch.Tensor:
        """(B, *feature_shape) in the cache dtype. Cast to float32 at the loss,
        not here -- fp16 MSE on unnormalized ViT features underflows to zero."""
        return self._gather(CLEAN_VIEW, self.rows_for(indices))

    def degraded(self, indices, epoch: int) -> torch.Tensor:
        return self._gather(view_name(epoch), self.rows_for(indices))

    def recipes(self, epoch: int) -> pd.DataFrame:
        """Per-image recipe table for one epoch, indexed by manifest index.

        Carries `severity`, the label-free conditioning target, and `transforms`,
        which is what per-transform recovery groups by.
        """
        return pd.read_parquet(self.root / view_name(epoch) / RECIPE_FILE).set_index("index")

    def severity(self, epoch: int) -> np.ndarray:
        """Severity per row, in row order, for the whole view."""
        table = self.recipes(epoch)
        return table.loc[self._index, "severity"].to_numpy(dtype=np.float32)

    def __getstate__(self) -> dict:
        """Never pickle the open memmaps.

        Under `spawn` -- macOS by default, Linux from 3.14 -- a DataLoader
        pickles the whole Dataset, and this cache travels inside it. Pickling
        `_shards` would either fail or, worse, succeed by materializing every
        mapped byte into the message. The worker re-opens its own handles
        lazily on first access, which is the invariant either way.
        """
        return {**self.__dict__, "_shards": {}}

    def worker_init(self, worker_id: int) -> None:
        """DataLoader `worker_init_fn`. Drops any inherited handles so this
        worker opens its own -- belt to `__getstate__`'s braces, and the one
        that matters under `fork`, where nothing is pickled."""
        self._shards = {}
