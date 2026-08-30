"""The crop, once it is wired into the render and the cache's identity.

Two things can go wrong here and neither shows up as an error:

  * the clean view and the degraded views are windows of *different* pixels, so
    stage 1 trains `adapter(f_deg) -> f_clean` between two unrelated pictures;
  * a cache of 128-512px windows is read by a run that thinks it holds whole
    images, so every feature is of something other than what the run believes.

`crop_sha` covers the second. The first is a property of `SAMPLE_EPOCH` and is
checked here directly, at the dataset, because by the time it reaches features
it is invisible.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from grace.cache.reader import FeatureCache
from grace.cache.schedule import EpochSchedule
from grace.cache.spec import CacheSpec, sha_manifest, sha_preprocess
from grace.cache.writer import MultiViewDataset, build_cache
from grace.config import CropConfig
from preprocessing.degrade.conditions import load_grid
from preprocessing.degrade.crop import SampleCrop
from tests.fixtures import SPECS, ToySplit

N_IMAGES = 12
GRID_FILE = "preprocessing/configs/degradations.yaml"
S_MIN, S_MAX = 32, 64


def write_images(directory, n: int, size: int = 96, seed: int = 0):
    """Bigger than `fixtures.write_images`' 32px, so a 32-64px window is a real
    choice of region rather than the whole picture every time."""
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        path = directory / f"{i:03d}.png"
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(path)
        rows.append({"path": str(path), "label": i % 2, "generator": "T", "split": "train"})
    return pd.DataFrame(rows)


def crop_cfg(**kw) -> CropConfig:
    return CropConfig(**{"enabled": True, "s_min": S_MIN, "s_max": S_MAX, **kw})


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """A real cache rendered through a multi-scale crop."""
    root = tmp_path_factory.mktemp("cropcache")
    manifest = write_images(root / "images", N_IMAGES)

    split = ToySplit(SPECS["vector"])
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    cfg = crop_cfg()
    spec = CacheSpec(
        detector="toy",
        feature=split.feature_spec,
        n=len(manifest),
        shard_size=10,
        manifest_sha=sha_manifest(manifest),
        schedule_sha=schedule.fingerprint(),
        detector_sha="toy",
        preprocess_sha=sha_preprocess(split.preprocess_fn()),
        crop_sha=cfg.fingerprint(),
    )
    out = root / "toy"
    build_cache(
        split, manifest, out, spec, schedule, [0, 1],
        batch_size=4, num_workers=0, crop=cfg.build(),
    )
    return out, manifest, split, schedule, spec, cfg


# --- the pairing property ----------------------------------------------------

class SpyCrop(SampleCrop):
    """Records every window it hands out, so the views can be compared."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list = []

    def __call__(self, img, index):
        self.calls.append((int(index), self.draw(img, index)))
        return super().__call__(img, index)


def test_every_view_of_an_image_is_the_same_window(tmp_path):
    """Stage 1 maps `f_deg` onto `f_clean` row by row. If the clean view were a
    different window, that is not a restoration task -- it is a request to
    hallucinate a region the input never showed."""
    manifest = write_images(tmp_path / "images", 4)
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    spy = SpyCrop(S_MIN, S_MAX, 0, "uniform")

    ds = MultiViewDataset(
        manifest, schedule, [None, 0, 1, 2], ToySplit(SPECS["vector"]).preprocess_fn(),
        crop=spy,
    )
    ds[0]

    assert len(spy.calls) == 4, "clean view included"
    indices = {i for i, _ in spy.calls}
    assert len(indices) == 1
    assert len({d for _, d in spy.calls}) == 1, "clean and degraded must share a window"


def test_different_images_get_different_windows(tmp_path):
    """The multi-scale diversity is across the corpus, since it cannot be across
    epochs -- see `preprocessing.degrade.crop.SAMPLE_EPOCH`."""
    manifest = write_images(tmp_path / "images", 24)
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    spy = SpyCrop(S_MIN, S_MAX, 0, "uniform")
    ds = MultiViewDataset(
        manifest, schedule, [None], ToySplit(SPECS["vector"]).preprocess_fn(), crop=spy
    )
    for i in range(len(manifest)):
        ds[i]
    assert len({d.size for _, d in spy.calls}) > 1
    assert len({(d.left, d.top) for _, d in spy.calls}) > 1


# --- the render is of the window it claims -----------------------------------

def test_cropped_clean_features_match_a_live_forward(rendered):
    """The cropped equivalent of `test_cache_alignment.test_clean_features_match_live`:
    pre-rendering through a crop has to equal cropping in the loop."""
    out, manifest, split, _, _, cfg = rendered
    cache = FeatureCache(out)
    crop = cfg.build()

    picked = list(manifest.index)[:8]
    cached = cache.clean(np.asarray(picked)).float()
    live = torch.stack([
        split.trunk(
            split.preprocess_fn()(
                crop(Image.open(manifest.loc[i, "path"]).convert("RGB"), int(i))
            ).unsqueeze(0)
        )[0]
        for i in picked
    ])
    cos = torch.nn.functional.cosine_similarity(cached.flatten(1), live.flatten(1), dim=1)
    assert float(cos.min()) > 0.999


def test_cropped_degraded_features_match_a_live_forward(rendered):
    """Degrade, then crop -- in that order. Cropping first would leave the
    recipe acting on a 32-64px window instead of the native image, and the
    parameter grid is calibrated for the latter."""
    out, manifest, split, schedule, _, cfg = rendered
    cache = FeatureCache(out)
    crop = cfg.build()

    picked = list(manifest.index)[:8]
    for epoch in (0, 1):
        cached = cache.degraded(np.asarray(picked), epoch).float()
        live = []
        for i in picked:
            img = Image.open(manifest.loc[i, "path"]).convert("RGB")
            img, _ = schedule.apply(img, int(i), epoch)
            img = crop(img, int(i))
            live.append(split.trunk(split.preprocess_fn()(img).unsqueeze(0))[0])
        cos = torch.nn.functional.cosine_similarity(
            cached.flatten(1), torch.stack(live).flatten(1), dim=1
        )
        assert float(cos.min()) > 0.999, f"epoch {epoch}"


def test_a_cropped_render_differs_from_a_whole_image_one(rendered):
    """If these matched, the crop was never applied and every test above would
    pass against a no-op."""
    out, manifest, split, _, _, _ = rendered
    cache = FeatureCache(out)
    picked = list(manifest.index)[:8]

    cached = cache.clean(np.asarray(picked)).float()
    whole = torch.stack([
        split.trunk(
            split.preprocess_fn()(
                Image.open(manifest.loc[i, "path"]).convert("RGB")
            ).unsqueeze(0)
        )[0]
        for i in picked
    ])
    assert not torch.allclose(cached, whole, atol=1e-3)


# --- crop_sha: the cache cannot be mistaken for another protocol -------------

def expect(spec: CacheSpec, crop_sha: str) -> CacheSpec:
    return CacheSpec(
        detector=spec.detector, feature=spec.feature, n=0, crop_sha=crop_sha
    )


def test_a_matching_crop_protocol_loads(rendered):
    out, _, _, _, spec, cfg = rendered
    assert FeatureCache(out, expect=expect(spec, cfg.fingerprint())).spec.n == N_IMAGES


def test_a_whole_image_run_refuses_a_cropped_cache(rendered):
    """The silent one. Nothing about the features' shape or dtype says they are
    windows, so without this the run reads them and trains."""
    out, _, _, _, spec, _ = rendered
    with pytest.raises(ValueError, match="crop_sha"):
        FeatureCache(out, expect=expect(spec, ""))


def test_a_cropped_run_refuses_a_whole_image_cache(tmp_path):
    manifest = write_images(tmp_path / "images", 4)
    split = ToySplit(SPECS["vector"])
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    spec = CacheSpec(
        detector="toy", feature=split.feature_spec, n=len(manifest),
        manifest_sha=sha_manifest(manifest), schedule_sha=schedule.fingerprint(),
    )
    out = tmp_path / "whole"
    build_cache(split, manifest, out, spec, schedule, [0], batch_size=2, num_workers=0)

    with pytest.raises(ValueError, match="crop_sha"):
        FeatureCache(out, expect=expect(spec, crop_cfg().fingerprint()))


def test_a_different_crop_range_is_refused(rendered):
    """Same manifest, same detector, same schedule -- different pixels."""
    out, _, _, _, spec, _ = rendered
    other = crop_cfg(s_max=S_MAX + 16).fingerprint()
    with pytest.raises(ValueError, match="crop_sha"):
        FeatureCache(out, expect=expect(spec, other))


def test_the_refusal_names_the_whole_image_case_readably(rendered):
    out, _, _, _, spec, _ = rendered
    with pytest.raises(ValueError, match="whole images"):
        FeatureCache(out, expect=expect(spec, ""))


# --- the config forces the range to be looked up -----------------------------

def test_enabling_the_crop_without_a_range_is_an_error():
    """The safe upper bound is a property of the corpus, so there is no default
    to fall back to: on `wildfake_test` a 128-512 range makes realized crop size
    a 0.9895-AUC classifier all by itself."""
    with pytest.raises(ValueError, match="audit_sizes"):
        CropConfig(enabled=True)


def test_a_disabled_crop_needs_no_range():
    cfg = CropConfig()
    assert cfg.build() is None
    assert cfg.fingerprint() == ""


def test_the_range_is_validated():
    with pytest.raises(ValueError, match="s_min"):
        CropConfig(enabled=True, s_min=64, s_max=32)
    with pytest.raises(ValueError, match="policy"):
        CropConfig(enabled=True, s_max=64, policy="gaussian")


def test_the_fingerprint_tracks_every_parameter():
    base = crop_cfg()
    assert base.fingerprint() == crop_cfg().fingerprint()
    for change in ({"s_min": 16}, {"s_max": 128}, {"seed": 1}, {"policy": "log_uniform"}):
        assert crop_cfg(**change).fingerprint() != base.fingerprint()
