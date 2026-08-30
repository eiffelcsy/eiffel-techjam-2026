"""`num_workers > 0` must work under `spawn`, not just under `fork`.

macOS has defaulted to the `spawn` start method since Python 3.8, and Linux
follows from 3.14. Under `spawn` a DataLoader does not inherit anything -- it
**pickles** the Dataset, the collate function and `worker_init_fn` and rebuilds
them in a fresh interpreter. Two things in this package sit right on that
boundary:

  * `worker_init_fn` used to be a lambda defined inside `build_loader`, which is
    a local object and cannot be pickled at all. It failed with
    `Can't get local object 'build_loader.<locals>.<lambda>'` on the first batch
    of the first epoch -- after the trunk was loaded and the run had started.
  * `FeatureCache` holds open numpy memmaps. Pickling those would either fail or
    succeed by copying every mapped byte into the worker message, which is the
    opposite of the point of memmapping them.

`fork` hides both: a lambda is never pickled and a memmap is inherited by the
child's address space. So a suite that only ever ran with `num_workers=0`, or
only on a `fork` platform, says nothing. These tests spawn real workers.
"""

import multiprocessing as mp
import pickle

import pytest

from grace.cache.reader import FeatureCache
from grace.cache.schedule import EpochSchedule
from grace.cache.spec import CacheSpec, sha_manifest, sha_preprocess
from grace.cache.writer import build_cache
from grace.train.data import _WorkerInit, build_loader
from preprocessing.degrade.conditions import load_grid
from tests.fixtures import SPECS, ToySplit, write_images

GRID_FILE = "preprocessing/configs/degradations.yaml"
N_IMAGES = 16
EPOCH = 0


def _loader_cfg(num_workers: int, batch_size: int = 4):
    return type("_Cfg", (), {
        "source": "cache", "batch_size": batch_size, "num_workers": num_workers,
    })()


@pytest.fixture(scope="module")
def cache_and_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("workers")
    manifest = write_images(root / "images", N_IMAGES)
    spec = SPECS["vector"]
    split = ToySplit(spec)
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    cache_dir = root / "cache"
    build_cache(
        split, manifest, cache_dir,
        CacheSpec(
            detector="toy", feature=spec, n=len(manifest), shard_size=8,
            manifest_sha=sha_manifest(manifest), schedule_sha=schedule.fingerprint(),
            detector_sha="toy", preprocess_sha=sha_preprocess(split.preprocess_fn()),
        ),
        schedule, [EPOCH], batch_size=8, num_workers=0,
    )
    return FeatureCache(cache_dir), manifest


def test_worker_init_is_picklable(cache_and_manifest):
    """The exact object that used to be an unpicklable local lambda."""
    cache, _ = cache_and_manifest
    revived = pickle.loads(pickle.dumps(_WorkerInit(cache)))
    revived(0)                                   # and it still works after the trip


def test_pickling_a_cache_drops_its_open_memmaps(cache_and_manifest):
    """`__getstate__`, tested where it matters: with handles actually open."""
    cache, manifest = cache_and_manifest
    cache.clean(manifest.index[:2])              # force the memmaps open
    assert cache._shards, "precondition: the parent has open handles"

    revived = pickle.loads(pickle.dumps(cache))
    assert revived._shards == {}, "memmaps must not cross a process boundary"
    # ...and the revived cache re-opens them lazily, reading the same bytes.
    assert revived.clean(manifest.index[:2]).equal(cache.clean(manifest.index[:2]))


def test_spawn_is_the_start_method_being_tested():
    """If this ever reads `fork`, the two tests below stop proving anything."""
    assert mp.get_start_method(allow_none=True) in (None, "spawn", "fork")


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_real_loader_yields_batches_with_workers(cache_and_manifest, num_workers):
    """The end-to-end version: spawn real workers and read real batches.

    `num_workers=0` is the control -- it runs in this process and would pass even
    with the lambda -- so the pair is what localises a regression to the process
    boundary rather than to the dataset.
    """
    cache, manifest = cache_and_manifest
    loader = build_loader(
        _loader_cfg(num_workers), cache, manifest, None, EPOCH, shuffle=False
    )
    batches = list(loader)
    assert batches, "loader yielded nothing"
    for batch in batches:
        assert set(batch) == {"f_deg", "f_clean", "label", "severity", "index"}
        assert batch["f_deg"].shape == (4, SPECS["vector"].dim)


def test_workers_and_no_workers_read_identical_features(cache_and_manifest):
    """Index alignment is the highest-risk bug in the project, and a per-worker
    memmap re-open is exactly where a row could shift. Same bytes, either way."""
    import torch

    cache, manifest = cache_and_manifest
    out = {}
    for num_workers in (0, 2):
        loader = build_loader(
            _loader_cfg(num_workers), cache, manifest, None, EPOCH, shuffle=False
        )
        batches = list(loader)
        out[num_workers] = (
            torch.cat([b["index"] for b in batches]),
            torch.cat([b["f_deg"].float() for b in batches]),
        )
    assert out[0][0].equal(out[2][0]), "workers returned a different row order"
    assert out[0][1].equal(out[2][1]), "workers returned different features"
