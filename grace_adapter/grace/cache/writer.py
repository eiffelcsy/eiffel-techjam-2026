"""Offline feature extraction: one trunk pass per (image, view), ever.

Layout on disk -- flat .npy memmaps in shards, not HDF5 or LMDB. The simplest
thing that supports random access and multi-worker reads.

    cache/{detector}/
    |-- spec.json               CacheSpec, incl. the four fingerprints
    |-- index.npy               int64, row -> manifest index (shared by all views)
    |-- .progress               how far the current multi-view pass has got
    |-- clean/
    |   |-- feats_00000.npy     (rows_in_shard, *feature_shape) float16
    |   `-- .done
    |-- epoch=000/
    |   |-- feats_00000.npy
    |   |-- recipes.parquet     index, level, recipe label, transforms, severity
    |   `-- .done
    |-- epoch=001/ ...
    `-- taps/                   ONLY when the split emits taps
        |-- clean/
        |   |-- feats_00000.npy (rows_in_shard, K, tap_dim) float16
        |   `-- .done
        `-- epoch=000/ ...

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

Images are the outer loop, views the inner
------------------------------------------
A DataLoader worker decodes an image **once** and returns every view of it,
degraded and preprocessed. The arrangement this replaced ran a full pass per
view, which decoded the whole dataset once per view: for the DINOv3 cache, 15
decodes of each of 277,643 images. Decoding is 7.4 ms an image and it was 13.5
ms of every 66.9 ms image-view, thrown away fourteen times out of fifteen.

Measured end to end, 500 NTIRE images (1.82 MP mean) x 15 views, RTX 3060 Ti:

    per-view decode, PIL in 8 workers      533 s    71.1 ms/image-view
    one decode, PIL in 8 workers (this)    203 s    27.1 ms/image-view
    one decode, degradation on CUDA        128 s    17.1 ms/image-view

So the degradation staying in the workers is a **deliberate trade, not the
fastest option**. Moving the eleven transforms onto the GPU is faster again --
they are 10-50x quicker per call there -- but it costs a second implementation
of every transform to keep, and getting one bit-identical to Pillow means
porting Pillow's own arithmetic: its three-pass extended box blur with its
float32 radius, `precompute_coeffs` in 22-bit fixed point, its integer
degenerate images. That was built, validated at >= 0.99998 feature cosine on ten
of eleven transforms, and then dropped as not worth carrying. If the render ever
needs the other 1.6x, that is the road, and it is a day of careful work.

Do not expect a GPU render to scale further than that 128 s: three concurrent
CUDA processes measured 11 ms -> 350 ms each (context thrash, no MPS on consumer
Windows), and batching same-shape images buys 1.0-1.1x because the ops are
bandwidth-bound rather than launch-bound.

Note also what the workers do *not* give you: eight of them scale this workload
about 2.4x, not 8x. PIL degradation on 2 MP images is memory-bound and the box
has six physical cores. Any model of this loop that divides single-threaded cost
by `num_workers` will be wrong by roughly 3x.

What the inversion costs is resume granularity: a view is no longer finished
before the next one starts, so a crash cannot be recovered by "skip the views
with a `.done`". `.progress` is the replacement -- see `_Progress`.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from grace.cache.schedule import EpochSchedule
from grace.cache.spec import DONE_FILE, INDEX_FILE, CacheSpec, tap_view_name, view_name
from grace.splits.base import SplitDetector
from pipeline.data.dataset import load_normalized

RECIPE_FILE = "recipes.parquet"
PROGRESS_FILE = ".progress"


class ShardWriter:
    """Append rows into fixed-size .npy shards, in order.

    Ordered append only -- no seeking, no partial rewrite. Shards are
    preallocated with `open_memmap`, so the header is valid from the first byte
    and a crashed render leaves a readable but unmarked directory.

    `start_row` resumes a view whose first `start_row` rows are already on disk
    in finished shards. It must land on a shard boundary, which is what
    `_Progress` records: a half-written shard is simply rewritten, since
    `open_memmap(mode="w+")` truncates.
    """

    def __init__(
        self,
        view_dir: str | Path,
        spec: CacheSpec,
        start_row: int = 0,
        feature=None,
    ):
        """`feature` overrides `spec.feature` for tap views, whose rows are
        `(K, tap_dim)` rather than the seam's shape. Everything else -- shards,
        resume, `.done` -- is identical, which is the point of storing taps as
        ordinary views instead of inventing a second on-disk format."""
        if start_row % spec.shard_size:
            raise ValueError(
                f"start_row {start_row} is not a multiple of shard_size "
                f"{spec.shard_size}; resuming mid-shard would leave a hole."
            )
        self.dir = Path(view_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.feature = feature if feature is not None else spec.feature
        self.row = start_row
        self._shard_id = -1
        self._shard = None

    def _rows_in(self, shard_id: int) -> int:
        start = shard_id * self.spec.shard_size
        return min(self.spec.shard_size, self.spec.n - start)

    def _ensure(self, shard_id: int):
        if shard_id != self._shard_id:
            self.flush()
            self._shard = np.lib.format.open_memmap(
                self.dir / f"feats_{shard_id:05d}.npy",
                mode="w+",
                dtype=np.dtype(self.feature.dtype),
                shape=(self._rows_in(shard_id), *self.feature.shape),
            )
            self._shard_id = shard_id
        return self._shard

    def write(self, features: np.ndarray) -> None:
        """Append a batch. Splits across a shard boundary when it straddles one."""
        if features.shape[1:] != self.feature.shape:
            raise ValueError(
                f"trunk emitted {features.shape[1:]}, spec declares "
                f"{self.feature.shape}"
            )
        written = 0
        while written < len(features):
            shard_id, offset = divmod(self.row, self.spec.shard_size)
            take = min(len(features) - written, self.spec.shard_size - offset)
            shard = self._ensure(shard_id)
            shard[offset : offset + take] = features[written : written + take]
            written += take
            self.row += take

    def flush(self) -> None:
        """Push the open shard's dirty pages to disk, keeping it open.

        Called whenever `_render` checkpoints, not only at the end: `_Progress`
        claims those rows are durable, and with fifteen views open at once an
        unflushed memmap per view is a lot of pages to lose.

        Keeping the memmap open is the point. `_ensure` opens shards with
        `mode="w+"`, which *truncates*, so a `flush` that dropped the handle
        would have the next write to the same shard re-create it and zero every
        row already in it -- silently, and only for shards a checkpoint happened
        to land inside.
        """
        if self._shard is not None:
            self._shard.flush()

    def finalize(self) -> None:
        if self.row != self.spec.n:
            raise RuntimeError(f"wrote {self.row} rows, expected {self.spec.n}")
        self.flush()
        (self.dir / DONE_FILE).write_text("ok", encoding="utf-8")


def is_complete(view_dir: str | Path) -> bool:
    return (Path(view_dir) / DONE_FILE).exists()


def view_is_complete(root: Path, epoch, spec: CacheSpec) -> bool:
    """A view counts as rendered only when every part of it is.

    With taps, one epoch is two directories. Testing the feature view alone
    would skip an epoch whose taps were never finished -- and the ladder would
    then train against whatever a half-written tap shard happens to hold, which
    is zeros, silently.
    """
    if not is_complete(root / view_name(epoch)):
        return False
    return not spec.taps or is_complete(root / tap_view_name(epoch))


class _Progress:
    """Row-level checkpoint for a multi-view pass.

    Views are rendered together, so `.done` markers only appear at the very end
    and cannot say how far an interrupted render got. This records the number of
    *complete shards* -- the granularity at which every open view has been
    flushed to disk -- alongside the exact set of views the pass was rendering.

    The view set is part of the record because resuming into a different set
    would interleave rows from two different passes. A mismatch restarts from
    row 0 rather than trying to be clever; it is a rare case, and the expensive
    one is being wrong about it.
    """

    def __init__(self, root: Path, views: list[str], shard_size: int):
        self.path = root / PROGRESS_FILE
        self.views = list(views)
        self.shard_size = shard_size

    def resume_row(self) -> int:
        if not self.path.exists():
            return 0
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 0
        if saved.get("views") != self.views or saved.get("shard_size") != self.shard_size:
            return 0
        return int(saved.get("shards_done", 0)) * self.shard_size

    def record(self, rows_done: int) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "views": self.views,
                    "shard_size": self.shard_size,
                    "shards_done": rows_done // self.shard_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class MultiViewDataset(Dataset):
    """One decode -> every requested view of that image, preprocessed.

    Returns `(V, *input_shape)` per item, `V` being the views this pass is
    rendering, in the order given. `epochs` holds `None` for the clean view;
    that is not a special mechanism, it is the epoch that has no condition.

    Everything expensive happens here, in the worker: decode, degrade,
    preprocess. The parent process only runs the trunk. That is the whole
    performance argument of this module -- see the module docstring -- and it is
    also why the item is the preprocessed 224px views rather than the
    full-resolution image: the same bytes cross the process boundary either way,
    but degrading in the parent would serialize what eight workers can do at
    once.
    """

    def __init__(self, manifest, schedule: EpochSchedule, epochs, preprocess, crop=None):
        self.paths = manifest["path"].tolist()
        # The manifest index, not the row position: it is the stable image
        # identity that seeds degradations, so it must survive subsetting.
        self.index = manifest.index.tolist()
        self.schedule = schedule
        self.epochs = list(epochs)
        self.preprocess = preprocess
        self.crop = crop

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        index = int(self.index[i])
        image = load_normalized(self.paths[i])
        views = []
        for epoch in self.epochs:
            # Degrade, then crop. The recipe keeps acting at native resolution,
            # so the parameter grid keeps the calibration every earlier number
            # was measured at; the crop is a view selection applied afterwards.
            view = image if epoch is None else self.schedule.apply(image, index, epoch)[0]
            # The same window for the clean view and every degraded epoch --
            # `pipeline.degrade.crop.SAMPLE_EPOCH` explains why the pairing
            # stage 1 trains on requires it.
            if self.crop is not None:
                view = self.crop(view, index)
            views.append(self.preprocess(view))
        return torch.stack(views), index


def collate_views(batch):
    """(B, V, *input_shape) and the manifest indices, in order."""
    views, indices = zip(*batch)
    return torch.stack(views), list(indices)


@torch.no_grad()
def build_cache(
    split: SplitDetector,
    manifest,
    root: str | Path,
    spec: CacheSpec,
    schedule: EpochSchedule,
    epochs,
    batch_size: int = 8,
    trunk_batch_size: int = 128,
    num_workers: int = 8,
    device=None,
    crop=None,
) -> CacheSpec:
    """Render the clean view plus every requested epoch, in a single pass.

    Resumable at shard granularity: rerun after an interruption and the pass
    picks up at the last checkpoint. A view that already carries `.done` from an
    earlier, completed render is skipped entirely, so adding epochs to a
    finished cache renders only the new ones.

    `crop` is an optional `(image, index) -> image` applied after the degradation
    recipe and before preprocessing -- see `pipeline.degrade.crop.SampleCrop`.
    Its identity belongs in `spec.crop_sha`, so features rendered under one
    window protocol can never be read as if they were another's.
    """
    split.assert_frozen()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    device = device or next(split.parameters()).device
    np.save(root / INDEX_FILE, np.asarray(manifest.index, dtype=np.int64))

    pending = [e for e in [None, *epochs] if not view_is_complete(root, e, spec)]
    if pending:
        rendering = [view_name(e) for e in pending]
        if spec.taps:
            rendering += [tap_view_name(e) for e in pending]
        progress = _Progress(root, rendering, spec.shard_size)
        start_row = min(progress.resume_row(), spec.n)
        if start_row:
            print(f"resuming at row {start_row} of {spec.n}")
        # Entered even when `start_row == spec.n`, which is a crash between the
        # last checkpoint and `finalize`: the rows are on disk but no view
        # carries `.done`, so the render is neither complete nor resumable.
        # `_render` then writes no rows and finalizes, which is exactly right.
        _render(
            split, manifest, root, spec, schedule, pending,
            progress=progress, start_row=start_row, batch_size=batch_size,
            trunk_batch_size=trunk_batch_size, num_workers=num_workers, device=device,
            crop=crop,
        )
        progress.clear()

    for epoch in epochs:
        _write_recipes(root / view_name(epoch), manifest, schedule, epoch)

    views = [None, *epochs]
    spec = replace(spec, views=tuple(view_name(e) for e in views))
    spec.save(root)
    return spec


def _render(
    split, manifest, root: Path, spec: CacheSpec, schedule, pending,
    progress: _Progress, start_row: int, batch_size: int, trunk_batch_size: int,
    num_workers: int, device, crop=None,
) -> None:
    """The single pass: decode each image once, render every pending view of it."""
    writers = [
        ShardWriter(root / view_name(epoch), spec, start_row=start_row)
        for epoch in pending
    ]
    # Taps ride along in the same pass. Rendering them separately would mean a
    # second full decode-degrade-forward over the dataset for activations the
    # first pass already computed and dropped -- which is the exact cost the
    # single-pass inversion above exists to avoid.
    tap_writers = [
        ShardWriter(
            root / tap_view_name(epoch), spec, start_row=start_row,
            feature=spec.tap_feature,
        )
        for epoch in pending
    ] if spec.taps else []

    # Slicing the manifest rather than the Dataset keeps row N of the cache at
    # manifest position N: the writers start at `start_row` and the loader must
    # hand them exactly the rows from there on, in order.
    dataset = MultiViewDataset(
        manifest.iloc[start_row:], schedule, pending, split.preprocess_fn(), crop=crop
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,                 # manifest order IS the row order. Never shuffle.
        collate_fn=collate_views,
        pin_memory=torch.device(device).type == "cuda",
    )

    rows = start_row
    # Checkpoint when a shard boundary is *crossed*, not when `rows` lands on
    # one: `batch_size` need not divide `shard_size`, and testing for equality
    # means a config where it does not never checkpoints at all -- an
    # interrupted render silently restarting from zero, hours in.
    recorded = start_row // spec.shard_size
    bar = tqdm(loader, desc=f"{root.name} x{len(writers)} views", total=len(loader))
    for views, _ in bar:
        images, n_views = views.shape[0], views.shape[1]
        # Flatten so the trunk sees `trunk_batch_size` samples regardless of how
        # few images a batch carries -- `batch_size` is small here because an
        # item is V preprocessed views, not one.
        flat = views.flatten(0, 1).to(device, non_blocking=True)
        chunks = [
            split.trunk_with_taps(flat[lo : lo + trunk_batch_size])
            for lo in range(0, len(flat), trunk_batch_size)
        ]
        features = torch.cat([f for f, _ in chunks]).view(
            images, n_views, *spec.feature.shape
        )
        taps = (
            torch.cat([t for _, t in chunks]).view(
                images, n_views, *spec.tap_feature.shape
            )
            if tap_writers
            else None
        )

        for view, writer in enumerate(writers):
            writer.write(features[:, view].to(spec.feature.torch_dtype).cpu().numpy())
        for view, writer in enumerate(tap_writers):
            writer.write(taps[:, view].to(spec.tap_feature.torch_dtype).cpu().numpy())

        rows += images
        if rows // spec.shard_size > recorded:
            for writer in (*writers, *tap_writers):
                writer.flush()
            # `record` floors to whole shards, so a batch that straddles a
            # boundary checkpoints only the shards behind it; the few rows past
            # it are simply re-rendered on resume.
            progress.record(rows)
            recorded = rows // spec.shard_size
        bar.set_postfix(rows=rows, refresh=False)

    for writer in (*writers, *tap_writers):
        writer.finalize()


def _write_recipes(view_dir: Path, manifest, schedule: EpochSchedule, epoch: int) -> None:
    """The per-image recipe table for one degraded view.

    Read back out of the schedule rather than accumulated during the render.
    The schedule is a pure function of (index, epoch) -- that is the property
    the whole offline cache rests on -- so this reproduces exactly what was
    applied, and shipping 4.16M recipe dicts back from the workers to avoid
    recomputing a blake2b hash would be a strange trade.
    """
    path = view_dir / RECIPE_FILE
    if path.exists():
        return
    rows = []
    for index in manifest.index:
        index = int(index)
        recipe = schedule.recipe_for(index, epoch)
        rows.append({
            "index": index,
            "level": schedule.level_for(index, epoch),
            "recipe": recipe.label(),
            "transforms": list(recipe.transforms()),
            "severity": schedule.severity_of(recipe),
        })
    pd.DataFrame(rows).to_parquet(path, index=False)
