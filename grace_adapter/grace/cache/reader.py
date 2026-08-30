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

from grace.cache.spec import (
    CLEAN_VIEW, INDEX_FILE, CacheSpec, freq_view_name, tap_view_name, view_name,
)
from grace.cache.writer import RECIPE_FILE, is_complete
from grace.splits.base import FeatureSpec


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
        # Which cached taps this reader hands back. Derived from `expect`
        # rather than taken as its own argument: the caller already states what
        # it needs there, and two places to say it is two places to disagree.
        self._tap_select = self._resolve_tap_select(expect)

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

    def _gather(self, name: str, rows: np.ndarray, feature=None) -> torch.Tensor:
        feature = feature if feature is not None else self._spec.feature
        shards = self._view(name)
        size = self._spec.shard_size
        out = np.empty((len(rows), *feature.shape), dtype=feature.dtype)
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

    @property
    def has_taps(self) -> bool:
        return bool(self._spec.taps) and self._spec.tap_feature is not None

    def _resolve_tap_select(self, expect: CacheSpec | None) -> list[int] | None:
        """Column indices into the cached tap axis, or None to hand back all.

        `assert_compatible` has already refused anything the cache does not
        hold, so this is pure lookup. Returns None -- not `range(len)` -- when
        the request is the whole set, so the common case does no indexing at all.

        Order follows the REQUEST, not the cache: the adapter's `tap_names` come
        from `SplitDetector.taps()`, and its per-tap gates are indexed by
        position, so a reordering here would silently attach every gate to the
        wrong block.
        """
        if expect is None or not expect.taps or not self.has_taps:
            return None
        want = tuple(expect.taps)
        if want == tuple(self._spec.taps):
            return None
        return [self._spec.taps.index(name) for name in want]

    @property
    def taps_selected(self) -> tuple[str, ...]:
        """The tap names this reader actually returns, in returned order."""
        if self._tap_select is None:
            return tuple(self._spec.taps)
        return tuple(self._spec.taps[i] for i in self._tap_select)

    @property
    def tap_feature(self) -> FeatureSpec | None:
        """Shape of what `taps()` RETURNS, after selection -- which is not
        `spec.tap_feature` when a subset is being read."""
        cached = self._spec.tap_feature
        if cached is None or self._tap_select is None:
            return cached
        return FeatureSpec(
            layout=cached.layout,
            shape=(len(self._tap_select), cached.dim),
            dtype=cached.dtype,
        )

    def _tap_gather(self, name: str, indices) -> torch.Tensor:
        if not self.has_taps:
            raise FileNotFoundError(
                f"{self.root} was rendered without taps, so the ladder has nothing "
                f"to read. Re-render with `split_args.tap_blocks` set "
                f"(scripts/build_cache.py), or train the plain adapter."
            )
        # Gathered at the cache's full width and then sliced. The rows come off
        # a memmap either way -- fancy-indexing the columns first would still
        # touch every byte of each row -- so this costs a view, not a read.
        out = self._gather(name, self.rows_for(indices), self._spec.tap_feature)
        return out if self._tap_select is None else out[:, self._tap_select]

    def clean_taps(self, indices) -> torch.Tensor:
        """`(B, K, tap_dim)`. The clean view of the taps, rendered by every tap
        cache. Not read by training -- it is here for analysis scripts that want
        the undamaged taps without re-rendering the cache."""
        return self._tap_gather(tap_view_name(None), indices)

    def taps(self, indices, epoch: int) -> torch.Tensor:
        return self._tap_gather(tap_view_name(epoch), indices)

    @property
    def has_freq(self) -> bool:
        return self._spec.freq_feature is not None

    def _freq_gather(self, name: str, indices) -> torch.Tensor:
        if not self.has_freq:
            raise FileNotFoundError(
                f"{self.root} was rendered without a frequency view, so the "
                f"enricher has nothing to read. Re-render with `freq.enabled: "
                f"true` (scripts/build_cache.py configs/cache/wildfake_freq.yaml), "
                f"or train the plain adapter."
            )
        return self._gather(name, self.rows_for(indices), self._spec.freq_feature)

    def clean_freq(self, indices) -> torch.Tensor:
        """`(B, n_cells, n_coeffs)` for the undegraded window.

        Read, unlike `clean_taps`: `analyze_freq.py` needs it to say what a band
        looks like before any transform touched it, which is the reference every
        per-degradation band signature is a delta from.
        """
        return self._freq_gather(freq_view_name(None), indices)

    def freq(self, indices, epoch: int) -> torch.Tensor:
        return self._freq_gather(freq_view_name(epoch), indices)

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
