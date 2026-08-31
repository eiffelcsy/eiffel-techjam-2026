"""The incremental append: grow a cache by reusing its shards and rendering only
the new rows.

The claim under test is the same one `test_cache_alignment.py` makes for a full
render, plus one more: an appended cache is byte-identical to a full re-render
of the grown manifest, and refuses to run if any fingerprint the features depend
on has moved.
"""

import pytest
import torch
from PIL import Image

from train.cache.reader import FeatureCache
from train.cache.schedule import EpochSchedule, val_epochs
from train.cache.spec import (
    CLEAN_VIEW, CacheSpec, assert_appendable, sha_manifest, sha_preprocess,
    view_name,
)
from train.cache.writer import append_cache, build_cache
from preprocessing.degrade.conditions import load_grid
from tests.fixtures import SPECS, ToySplit, write_images

N_IMAGES = 24
MORE = 12
GRID_FILE = "preprocessing/configs/degradations.yaml"


def _make_spec(split, manifest, schedule, shard_size=10):
    return CacheSpec(
        detector="toy",
        feature=split.feature_spec,
        n=len(manifest),
        shard_size=shard_size,
        manifest_sha=sha_manifest(manifest),
        schedule_sha=schedule.fingerprint(),
        detector_sha="toy",
        preprocess_sha=sha_preprocess(split.preprocess_fn()),
    )


@pytest.fixture(scope="module")
def grown(tmp_path_factory):
    """An old 24-row cache, appended to 36 rows, beside a full 36-row reference.

    `write_images` is seeded and deterministic, so re-calling it for 36 images
    regenerates the first 24 bytes identically -- the combined manifest is an
    exact prefix of the old one, which is the situation the append is for.
    """
    root = tmp_path_factory.mktemp("cache")
    img_dir = root / "images"
    old_manifest = write_images(img_dir, N_IMAGES)

    split = ToySplit(SPECS["layers"])
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    epochs = [0, 1, *val_epochs(1)]

    out = root / "toy"
    build_cache(split, old_manifest, out, _make_spec(split, old_manifest, schedule),
                schedule, epochs, batch_size=4, num_workers=0)
    old_spec = CacheSpec.load(out)          # the on-disk spec, `views` included

    reused = {
        (view, shard): (out / view / shard).stat().st_mtime_ns
        for view in (CLEAN_VIEW, view_name(0))
        for shard in ("feats_00000.npy", "feats_00001.npy")
    }

    combined = write_images(img_dir, N_IMAGES + MORE)
    new_spec = _make_spec(split, combined, schedule)
    old_n = assert_appendable(old_spec, new_spec, combined, epochs)
    append_cache(split, combined, out, new_spec, schedule, epochs, old_n,
                 batch_size=4, num_workers=0)

    reference = root / "reference"
    build_cache(split, combined, reference, new_spec, schedule, epochs,
                batch_size=4, num_workers=0)

    return {
        "out": out, "reference": reference, "combined": combined,
        "split": split, "schedule": schedule, "epochs": epochs,
        "new_spec": new_spec, "old_spec": old_spec,
        "reused": reused, "old_n": old_n,
    }


def test_reused_shards_are_untouched(grown):
    """Shards fully below the append boundary keep their bytes -- and mtimes."""
    out, reused = grown["out"], grown["reused"]
    for (view, shard), before in reused.items():
        assert (out / view / shard).stat().st_mtime_ns == before


def test_append_matches_full_rerender(grown):
    """The gold standard: appended == re-rendered from scratch, bit for bit."""
    out, reference, combined = grown["out"], grown["reference"], grown["combined"]
    a, b = FeatureCache(out), FeatureCache(reference)
    assert torch.equal(a.clean(combined.index), b.clean(combined.index))
    for epoch in (0, 1):
        assert torch.equal(
            a.degraded(combined.index, epoch), b.degraded(combined.index, epoch)
        )


def test_appended_features_match_live(grown):
    """The new rows are features of the new images, not garbage."""
    out, combined, split = grown["out"], grown["combined"], grown["split"]
    cache = FeatureCache(out)
    new_idx = combined.index[N_IMAGES:]
    cached = cache.clean(new_idx).float()
    live = torch.stack([
        split.trunk(split.preprocess_fn()(Image.open(combined.loc[i, "path"])).unsqueeze(0))[0]
        for i in new_idx
    ])
    cos = torch.nn.functional.cosine_similarity(cached.flatten(1), live.flatten(1), dim=1)
    assert float(cos.min()) > 0.999


def test_recipes_cover_new_rows(grown):
    out, combined, schedule = grown["out"], grown["combined"], grown["schedule"]
    table = FeatureCache(out).recipes(0)
    assert len(table) == len(combined)
    for i in combined.index[N_IMAGES:]:
        assert table.loc[int(i), "severity"] == pytest.approx(
            schedule.severity_for(int(i), 0)
        )


def test_epochs_still_reports_finished_views(grown):
    assert set(FeatureCache(grown["out"]).epochs()) == {0, 1, min(val_epochs(1))}


def test_append_when_existing_is_smaller_than_one_shard(tmp_path):
    """The partial-shard path: `old_n` smaller than one shard is re-shaped (the
    existing rows are copied, not re-rendered) and the result must match a fresh
    render of the grown manifest."""
    root = tmp_path / "cache"
    img_dir = root / "images"
    old_manifest = write_images(img_dir, 8)
    split = ToySplit(SPECS["layers"])
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    epochs = [0, 1, *val_epochs(1)]

    out = root / "toy"
    build_cache(split, old_manifest, out, _make_spec(split, old_manifest, schedule),
                schedule, epochs, batch_size=4, num_workers=0)
    old_spec = CacheSpec.load(out)

    combined = write_images(img_dir, 16)
    new_spec = _make_spec(split, combined, schedule)
    old_n = assert_appendable(old_spec, new_spec, combined, epochs)
    assert old_n == 8
    append_cache(split, combined, out, new_spec, schedule, epochs, old_n,
                 batch_size=4, num_workers=0)

    reference = root / "reference"
    build_cache(split, combined, reference, new_spec, schedule, epochs,
                batch_size=4, num_workers=0)
    a, b = FeatureCache(out), FeatureCache(reference)
    assert torch.equal(a.clean(combined.index), b.clean(combined.index))


# --- the enforcement ----------------------------------------------------------


def test_rejects_nothing_to_append(grown):
    with pytest.raises(ValueError, match="nothing to append"):
        assert_appendable(grown["new_spec"], grown["new_spec"], grown["combined"],
                          grown["epochs"])


def test_rejects_a_changed_fingerprint(grown):
    from dataclasses import replace

    bad = replace(grown["new_spec"], schedule_sha="different")
    with pytest.raises(ValueError, match="schedule_sha"):
        assert_appendable(grown["old_spec"], bad, grown["combined"], grown["epochs"])


def test_rejects_a_changed_view_set(grown):
    # Appending with a different epoch list is silent corruption: the reused rows
    # of a dropped epoch vanish, and an added epoch has no reused rows to reuse.
    with pytest.raises(ValueError, match="view set"):
        assert_appendable(grown["old_spec"], grown["new_spec"], grown["combined"],
                          [0, 1, 2, *val_epochs(1)])


def test_rejects_a_non_prefix_manifest(grown):
    combined, split, schedule = (
        grown["combined"], grown["split"], grown["schedule"]
    )
    altered = combined.copy()
    altered.loc[altered.index[0], "label"] = 1 - altered.loc[altered.index[0], "label"]
    with pytest.raises(ValueError, match="exact prefix"):
        assert_appendable(
            grown["old_spec"], _make_spec(split, altered, schedule), altered,
            grown["epochs"],
        )


def test_rejects_a_shard_size_change(grown):
    spec = _make_spec(grown["split"], grown["combined"], grown["schedule"], shard_size=20)
    with pytest.raises(ValueError, match="shard_size"):
        assert_appendable(grown["old_spec"], spec, grown["combined"], grown["epochs"])


def test_rejects_a_feature_change(grown):
    other = ToySplit(SPECS["vector"])
    spec = _make_spec(other, grown["combined"], grown["schedule"])
    with pytest.raises(ValueError, match="feature spec"):
        assert_appendable(grown["old_spec"], spec, grown["combined"], grown["epochs"])


def test_append_cache_refuses_a_non_positive_old_n(grown):
    with pytest.raises(ValueError, match="old_n"):
        append_cache(
            grown["split"], grown["combined"], grown["out"], grown["new_spec"],
            grown["schedule"], grown["epochs"], 0, batch_size=4, num_workers=0,
        )
