"""The highest-risk bug in the project: row `i` of the cache is a different image
than row `i` of the dataset. It trains, it converges, it means nothing.

These run a real render end-to-end -- writer, spec, schedule, reader -- against a
toy split and a handful of generated images, so the plumbing is exercised without
needing any vendored detector weights.
"""

import json
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from train.cache.reader import FeatureCache
from train.cache.schedule import EpochSchedule, val_epochs
from train.cache.spec import (
    CLEAN_VIEW,
    DONE_FILE,
    CacheSpec,
    sha_manifest,
    sha_preprocess,
    view_name,
)
from train.cache.writer import build_cache
from preprocessing.degrade.conditions import load_grid
from tests.fixtures import SPECS, ToySplit, write_images

N_IMAGES = 24
GRID_FILE = "preprocessing/configs/degradations.yaml"


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """A real two-epoch cache over generated images."""
    root = tmp_path_factory.mktemp("cache")
    manifest = write_images(root / "images", N_IMAGES)

    split = ToySplit(SPECS["layers"])
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    epochs = [0, 1, *val_epochs(1)]
    spec = CacheSpec(
        detector="toy",
        feature=split.feature_spec,
        n=len(manifest),
        shard_size=10,                       # forces a multi-shard render
        manifest_sha=sha_manifest(manifest),
        schedule_sha=schedule.fingerprint(),
        detector_sha="toy",
        preprocess_sha=sha_preprocess(split.preprocess_fn()),
    )
    out = root / "toy"
    build_cache(split, manifest, out, spec, schedule, epochs, batch_size=4, num_workers=0)
    return out, manifest, split, schedule, spec


def test_clean_features_match_live(rendered):
    """20 random indices, trunk run live on the clean image, against the cache."""
    out, manifest, split, _, _ = rendered
    cache = FeatureCache(out)
    picked = np.random.default_rng(1).choice(manifest.index, size=20, replace=False)

    cached = cache.clean(picked).float()
    live = torch.stack([
        split.trunk(split.preprocess_fn()(Image.open(manifest.loc[i, "path"])).unsqueeze(0))[0]
        for i in picked
    ])
    cos = torch.nn.functional.cosine_similarity(cached.flatten(1), live.flatten(1), dim=1)
    assert float(cos.min()) > 0.999


def test_degraded_features_match_live(rendered):
    """The check that certifies pre-rendering as equivalent to degrading in the
    loop -- the whole architecture rests on this."""
    out, manifest, split, schedule, _ = rendered
    cache = FeatureCache(out)
    picked = np.random.default_rng(2).choice(manifest.index, size=12, replace=False)

    for epoch in (0, 1):
        cached = cache.degraded(picked, epoch).float()
        live = []
        for i in picked:
            img = Image.open(manifest.loc[i, "path"]).convert("RGB")
            img, _ = schedule.apply(img, int(i), epoch)
            live.append(split.trunk(split.preprocess_fn()(img).unsqueeze(0))[0])
        cos = torch.nn.functional.cosine_similarity(
            cached.flatten(1), torch.stack(live).flatten(1), dim=1
        )
        assert float(cos.min()) > 0.999, f"epoch {epoch}"


def test_clean_and_degraded_share_a_row(rendered):
    """One manifest index -> the same row in every view. That is what makes a
    (f_deg, f_clean) pair a single lookup."""
    out, manifest, _, _, _ = rendered
    cache = FeatureCache(out)
    rows = cache.rows_for(manifest.index)
    assert np.array_equal(cache.index[rows], np.asarray(manifest.index))


def test_survives_subsetting(rendered):
    """Indexing by manifest index must still land on the right row after the
    manifest is filtered -- row position would not."""
    out, manifest, _, _, _ = rendered
    cache = FeatureCache(out)
    subset = manifest.iloc[::3]
    full = cache.clean(manifest.index)
    assert torch.equal(cache.clean(subset.index), full[np.arange(len(manifest))[::3]])


def test_unknown_index_raises(rendered):
    out, _, _, _, _ = rendered
    with pytest.raises(KeyError, match="not in this cache"):
        FeatureCache(out).clean([999_999])


def test_severity_column_is_written(rendered):
    out, manifest, _, schedule, _ = rendered
    cache = FeatureCache(out)
    table = cache.recipes(0)
    assert len(table) == len(manifest)
    for i in list(manifest.index)[:5]:
        assert table.loc[i, "severity"] == pytest.approx(schedule.severity_for(int(i), 0))


def test_epochs_reports_only_finished_views(rendered):
    out, _, _, _, _ = rendered
    assert set(FeatureCache(out).epochs()) == {0, 1, min(val_epochs(1))}


def test_rejects_stale_manifest(rendered):
    """A changed manifest_sha must raise naming the manifest, not fail silently."""
    out, _, _, _, spec = rendered
    from dataclasses import replace

    with pytest.raises(ValueError, match="manifest_sha"):
        FeatureCache(out, expect=replace(spec, manifest_sha="different"))


def test_rejects_changed_schedule(rendered):
    out, _, _, _, spec = rendered
    from dataclasses import replace

    with pytest.raises(ValueError, match="schedule_sha"):
        FeatureCache(out, expect=replace(spec, schedule_sha="different"))


def test_accepts_a_matching_spec(rendered):
    out, _, _, _, spec = rendered
    assert FeatureCache(out, expect=spec).spec.n == N_IMAGES


def test_stochastic_preprocess_is_rejected():
    """A random crop makes the clean cache non-reproducible; fail at startup
    rather than 40 GB later."""
    class _Random:
        def __call__(self, img):
            return torch.randn(3, 8, 8)

    with pytest.raises(ValueError, match="not deterministic"):
        sha_preprocess(_Random())


# --- resuming an interrupted multi-view render --------------------------------
# Views are rendered together now, so `.done` markers only appear at the very
# end and cannot say how far an interrupted render got. `.progress` is what
# replaces them, and a checkpoint that is wrong by one shard produces a cache
# with a hole in it that nothing downstream would notice.

def test_shard_writer_refuses_a_mid_shard_resume():
    from train.cache.writer import ShardWriter

    spec = CacheSpec(detector="toy", feature=SPECS["vector"], n=20, shard_size=10)
    with pytest.raises(ValueError, match="not a multiple of shard_size"):
        ShardWriter("unused", spec, start_row=15)


def test_progress_checkpoints_when_batches_straddle_a_shard(rendered, tmp_path):
    """`batch_size` need not divide `shard_size`.

    Testing `rows % shard_size == 0` looks equivalent and is not: with a batch
    that steps over the boundary rather than onto it, the condition is never
    true and the render checkpoints exactly never. Mid-shard checkpoints in turn
    are what make `ShardWriter.flush` keeping its memmap open load-bearing --
    dropping it re-creates the shard with `mode="w+"` and zeroes the rows
    already in it.
    """
    from train.cache.writer import PROGRESS_FILE, build_cache

    out, manifest, split, schedule, spec = rendered
    epochs = [0, 1, *val_epochs(1)]
    root = tmp_path / "straddle"
    # 24 rows, shards of 10, batches of 7: boundaries fall at 10 and 20, batch
    # ends at 7, 14, 21, 24. Nothing lands on a boundary.
    build_cache(split, manifest, root, spec, schedule, epochs,
                batch_size=7, trunk_batch_size=16, num_workers=0)
    assert not (root / PROGRESS_FILE).exists()      # cleared on success
    reference = FeatureCache(out)
    assert torch.equal(FeatureCache(root).clean(manifest.index),
                       reference.clean(manifest.index))


def test_progress_ignores_a_checkpoint_from_a_different_view_set(tmp_path):
    """Resuming into a different set of views would interleave two passes."""
    from train.cache.writer import _Progress

    _Progress(tmp_path, ["clean", "epoch=000"], 10).record(30)
    assert _Progress(tmp_path, ["clean", "epoch=000"], 10).resume_row() == 30
    assert _Progress(tmp_path, ["clean", "epoch=001"], 10).resume_row() == 0
    assert _Progress(tmp_path, ["clean", "epoch=000"], 20).resume_row() == 0


def test_an_interrupted_render_resumes_to_the_same_features(rendered, tmp_path):
    """Truncate a finished cache back to a checkpoint, re-render, compare.

    Stands in for a crash: the shards past the checkpoint are gone, no view
    carries `.done`, and `.progress` says how far the pass got. What comes out
    has to be indistinguishable from the uninterrupted render.
    """
    from train.cache.writer import PROGRESS_FILE, build_cache

    out, manifest, split, schedule, spec = rendered
    reference = FeatureCache(out)
    epochs = [0, 1, *val_epochs(1)]

    partial = tmp_path / "partial"
    shutil.copytree(out, partial)
    kept_shards = 1                       # of N_IMAGES/shard_size = 24/10 -> 3
    for view in [CLEAN_VIEW, *(view_name(e) for e in epochs)]:
        (partial / view / DONE_FILE).unlink()
        for shard in sorted((partial / view).glob("feats_*.npy"))[kept_shards:]:
            shard.unlink()
    (partial / PROGRESS_FILE).write_text(
        json.dumps({
            "views": [CLEAN_VIEW, *(view_name(e) for e in epochs)],
            "shard_size": spec.shard_size,
            "shards_done": kept_shards,
        }),
        encoding="utf-8",
    )

    build_cache(split, manifest, partial, spec, schedule, epochs,
                batch_size=4, num_workers=0)

    resumed = FeatureCache(partial)
    assert not (partial / PROGRESS_FILE).exists()
    for view, getter in [
        ("clean", lambda c: c.clean(manifest.index)),
        ("epoch 0", lambda c: c.degraded(manifest.index, 0)),
        ("epoch 1", lambda c: c.degraded(manifest.index, 1)),
    ]:
        got, want = getter(resumed), getter(reference)
        if not torch.equal(got, want):
            # Name the rows: a bad checkpoint leaves a contiguous band wrong,
            # which says immediately whether the resume offset or the render is
            # at fault. A bare `assert` on two 24x4x16 tensors says neither.
            differing = (got.view(torch.int16) != want.view(torch.int16))
            rows = differing.flatten(1).any(1).nonzero().flatten().tolist()
            raise AssertionError(
                f"{view}: rows {rows} differ after resuming at row "
                f"{kept_shards * spec.shard_size}"
            )


def test_a_finished_view_is_not_re_rendered(rendered, tmp_path):
    """Adding an epoch to a finished cache must not redo the ones it has."""
    from train.cache.writer import build_cache

    out, manifest, split, schedule, spec = rendered
    grown = tmp_path / "grown"
    shutil.copytree(out, grown)
    before = (grown / CLEAN_VIEW / "feats_00000.npy").stat().st_mtime_ns

    build_cache(split, manifest, grown, spec, schedule, [0, 1, 2, *val_epochs(1)],
                batch_size=4, num_workers=0)

    assert (grown / CLEAN_VIEW / "feats_00000.npy").stat().st_mtime_ns == before
    assert set(FeatureCache(grown).epochs()) == {0, 1, 2, min(val_epochs(1))}
