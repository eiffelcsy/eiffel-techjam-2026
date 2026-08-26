"""The highest-risk bug in the project: row `i` of the cache is a different image
than row `i` of the dataset. It trains, it converges, it means nothing.

These run a real render end-to-end -- writer, spec, schedule, reader -- against a
toy split and a handful of generated images, so the plumbing is exercised without
needing any vendored detector weights.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from grace.cache.reader import FeatureCache
from grace.cache.schedule import EpochSchedule, val_epochs
from grace.cache.spec import CacheSpec, sha_manifest, sha_preprocess
from grace.cache.writer import build_cache
from pipeline.degrade.conditions import load_grid
from tests.fixtures import SPECS, ToySplit, write_images

N_IMAGES = 24
GRID_FILE = "../eval_pipeline/configs/degradations.yaml"


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
