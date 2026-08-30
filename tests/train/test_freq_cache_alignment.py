"""The frequency view is a third view family, and it fails the same way the
first two do: row `i` of `freq/` is a different image than row `i` of `clean/`.

The bug is silent -- the enricher would attend over another picture's spectrum
and still train to a plausible loss -- so these run a real render end to end
(writer, spec, schedule, reader) against a toy split and generated images, and
compare what came back against what `extract_freq` produces live. Mirrors
`test_cache_alignment.py`, deliberately: the two view families are the same
mechanism under two names, and the tests should look it.
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
    CLEAN_VIEW, DONE_FILE, CacheSpec, freq_view_name, sha_manifest, sha_preprocess,
    view_name,
)
from train.cache.writer import build_cache
from train.config import FreqConfig
from preprocessing.dataset import load_normalized
from preprocessing.degrade.conditions import load_grid
from freq_branch.dct import extract_freq
from tests.fixtures import SPECS, ToySplit, write_images

N_IMAGES = 24
GRID_FILE = "preprocessing/configs/degradations.yaml"

# A 2x2 block over a 3x3 grid: 36 coefficients per cell, 9 cells. Tiny, so the
# render is fast, and structurally identical to the shipped 8x8 / 14x14 -- what
# these tests check is which rows and which views the bytes came from, and that
# is not a function of the block size.
FREQ = FreqConfig(enabled=True, patch=2, grid=3)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp("freqcache")
    manifest = write_images(root / "images", N_IMAGES)

    split = ToySplit(SPECS["vector"])
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
        freq_feature=FREQ.feature(),
        freq_sha=FREQ.fingerprint(),
    )
    out = root / "toy"
    build_cache(
        split, manifest, out, spec, schedule, epochs,
        batch_size=4, num_workers=0, freq=FREQ.build(),
    )
    return out, manifest, split, schedule, spec


def _live_freq(path, schedule=None, index=None, epoch=None) -> torch.Tensor:
    img = load_normalized(path)
    if schedule is not None:
        img, _ = schedule.apply(img, int(index), epoch)
    return torch.from_numpy(
        extract_freq(np.asarray(img, dtype=np.uint8), FREQ.patch, FREQ.grid)
    )


def test_the_view_has_the_declared_shape(rendered):
    out, manifest, _, _, _ = rendered
    cache = FeatureCache(out)
    assert cache.has_freq
    assert cache.clean_freq(manifest.index[:4]).shape == (4, *FREQ.shape)


def test_clean_freq_matches_live(rendered):
    """The rendered clean view against `extract_freq` run here, now.

    Exact equality up to the fp16 the cache stores, not a cosine: the extractor
    is pure numpy with no model in it, so anything other than a rounding
    difference means the render read different pixels.
    """
    out, manifest, _, _, _ = rendered
    cache = FeatureCache(out)
    picked = np.random.default_rng(1).choice(manifest.index, size=8, replace=False)
    cached = cache.clean_freq(picked).float()
    live = torch.stack([_live_freq(manifest.loc[i, "path"]) for i in picked])
    assert torch.allclose(cached, live, atol=2e-3)


def test_degraded_freq_matches_live(rendered):
    """The property pre-rendering rests on, for the frequency side: epoch `e`'s
    view is a pure function of (image, epoch) and can be recomputed at will."""
    out, manifest, _, schedule, _ = rendered
    cache = FeatureCache(out)
    picked = np.random.default_rng(2).choice(manifest.index, size=6, replace=False)
    for epoch in (0, 1):
        cached = cache.freq(picked, epoch).float()
        live = torch.stack(
            [_live_freq(manifest.loc[i, "path"], schedule, i, epoch) for i in picked]
        )
        assert torch.allclose(cached, live, atol=2e-3), f"epoch {epoch}"


def test_degradation_actually_moved_the_spectrum(rendered):
    """A guard against the test above passing because both sides are the clean
    view: if the schedule were a no-op, `freq` and `clean_freq` would agree and
    every alignment check here would be vacuous."""
    out, manifest, _, _, _ = rendered
    cache = FeatureCache(out)
    clean = cache.clean_freq(manifest.index).float()
    degraded = cache.freq(manifest.index, 0).float()
    assert not torch.allclose(clean, degraded, atol=1e-2)


def test_freq_shares_a_row_with_the_features(rendered):
    """One manifest index -> the same row in `clean/`, `epoch=000/` and
    `freq/epoch=000/`. That is what makes (f_deg, freq_deg) a single lookup, and
    it is the whole reason both live in one cache root."""
    out, manifest, _, _, _ = rendered
    cache = FeatureCache(out)
    subset = manifest.iloc[::5]
    positions = np.arange(len(manifest))[::5]
    for getter in (
        lambda c, ix: c.clean_freq(ix),
        lambda c, ix: c.freq(ix, 0),
        lambda c, ix: c.degraded(ix, 0),
    ):
        assert torch.equal(getter(cache, subset.index), getter(cache, manifest.index)[positions])


def test_a_cache_without_the_view_refuses_rather_than_returning_zeros(tmp_path):
    out, manifest, split, schedule, _ = _plain_render(tmp_path)
    cache = FeatureCache(out)
    assert not cache.has_freq
    with pytest.raises(FileNotFoundError, match="without a frequency view"):
        cache.clean_freq(manifest.index[:2])


def test_assert_freq_available_names_the_protocol_mismatch(rendered):
    """Same shape, different frequencies is the mismatch worth refusing: the
    enricher's band masks are indexed by position along the coefficient axis, so
    a 4x1 render and a 2x2 render would both load and mean different things."""
    out, _, _, _, _ = rendered
    spec = FeatureCache(out).spec
    spec.assert_freq_available(FREQ.feature(), FREQ.fingerprint())
    with pytest.raises(ValueError, match="freq_sha differs"):
        spec.assert_freq_available(FREQ.feature(), "not-the-same-protocol")


def test_view_is_incomplete_until_the_freq_view_finishes(rendered, tmp_path):
    """One epoch is three directories now. A `.done` on the features alone must
    not count, or the render would skip an epoch whose frequency shards are
    half-written -- which is zeros, silently."""
    from train.cache.writer import view_is_complete

    out, _, _, _, spec = rendered
    partial = tmp_path / "partial"
    shutil.copytree(out, partial)
    assert view_is_complete(partial, 0, spec)
    (partial / freq_view_name(0) / DONE_FILE).unlink()
    assert not view_is_complete(partial, 0, spec)


def test_an_interrupted_render_resumes_to_the_same_freq_view(rendered, tmp_path):
    """Truncate every view back to one shard, re-render, compare.

    The frequency writers are checkpointed and flushed alongside the feature
    writers, so a resume that got the offset wrong on one family and not the
    other would leave a contiguous band of rows holding another image's spectrum
    -- which nothing downstream would notice.
    """
    from train.cache.writer import PROGRESS_FILE

    out, manifest, split, schedule, spec = rendered
    reference = FeatureCache(out)
    epochs = [0, 1, *val_epochs(1)]
    views = [CLEAN_VIEW, *(view_name(e) for e in epochs)]
    freq_views = [freq_view_name(e) for e in (None, *epochs)]

    partial = tmp_path / "resumed"
    shutil.copytree(out, partial)
    kept_shards = 1
    for view in views + freq_views:
        (partial / view / DONE_FILE).unlink()
        for shard in sorted((partial / view).glob("feats_*.npy"))[kept_shards:]:
            shard.unlink()
    (partial / PROGRESS_FILE).write_text(
        json.dumps({
            "views": views + freq_views,
            "shard_size": spec.shard_size,
            "shards_done": kept_shards,
        }),
        encoding="utf-8",
    )

    build_cache(
        split, manifest, partial, spec, schedule, epochs,
        batch_size=4, num_workers=0, freq=FREQ.build(),
    )
    resumed = FeatureCache(partial)
    for name, getter in [
        ("freq/clean", lambda c: c.clean_freq(manifest.index)),
        ("freq/epoch 0", lambda c: c.freq(manifest.index, 0)),
        ("epoch 0", lambda c: c.degraded(manifest.index, 0)),
    ]:
        got, want = getter(resumed), getter(reference)
        if not torch.equal(got, want):
            rows = (got != want).flatten(1).any(1).nonzero().flatten().tolist()
            raise AssertionError(
                f"{name}: rows {rows} differ after resuming at row "
                f"{kept_shards * spec.shard_size}"
            )


def test_a_spec_and_an_extractor_must_agree_at_render_time(tmp_path):
    """A spec claiming a view nothing is going to write produces a cache whose
    reader fails only when the enricher first asks for it -- hours later, in
    another script."""
    root = tmp_path / "mismatch"
    manifest = write_images(root / "images", 4)
    split = ToySplit(SPECS["vector"])
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    spec = CacheSpec(
        detector="toy", feature=split.feature_spec, n=len(manifest),
        freq_feature=FREQ.feature(), freq_sha=FREQ.fingerprint(),
    )
    with pytest.raises(ValueError, match="spec.freq_feature"):
        build_cache(split, manifest, root / "toy", spec, schedule, [0], num_workers=0)


def _plain_render(tmp_path):
    """A render with no frequency view -- the pre-freq layout, unchanged."""
    manifest = write_images(tmp_path / "images", 8)
    split = ToySplit(SPECS["vector"])
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    spec = CacheSpec(
        detector="toy", feature=split.feature_spec, n=len(manifest), shard_size=10,
    )
    out = tmp_path / "plain"
    build_cache(split, manifest, out, spec, schedule, [0], batch_size=4, num_workers=0)
    return out, manifest, split, schedule, spec


def test_a_render_without_freq_writes_the_original_layout(tmp_path):
    """The default path, byte for byte. A cache with no frequency view must have
    no `freq/` directory at all -- not an empty one, not one of zeros."""
    out, _, _, _, _ = _plain_render(tmp_path)
    assert not (out / "freq").exists()
    assert json.loads((out / "spec.json").read_text())["freq_feature"] is None


def test_a_freq_render_is_larger_by_exactly_the_view(rendered):
    """`bytes_per_view` is what `--dry-run` prints before a multi-hour render
    commits, and the frequency view is ~49x the features on the real detector --
    so an accounting that forgot it would under-report the disk cost by 98%."""
    out, _, _, _, spec = rendered
    features = spec.feature.bytes_per_image()
    assert spec.bytes_per_view() == features + spec.freq_feature.bytes_per_image()
