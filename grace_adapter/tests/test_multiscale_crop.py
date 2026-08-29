"""The crop is the cache's contract, the same way the schedule is.

Pre-rendering features of a *random* window is only sound because
`(index, epoch, seed)` determines the window exactly, on any machine, in any
process. These tests are what license that claim, plus the two properties the
benchmark repair rests on: the evaluation arms draw nothing, and a crop that had
to be shrunk to fit says so.
"""

import numpy as np
import pytest
from PIL import Image

from pipeline.degrade.crop import (
    CropDraw, crop_fingerprint, fixed_crop, fixed_resample, multiscale_crop,
)


def image(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mode="RGB")


@pytest.fixture(scope="module")
def big():
    return image(1024, 1024, seed=1)


# --- determinism -------------------------------------------------------------

def test_crop_is_pure(big):
    for index in (0, 7, 4242):
        for epoch in (0, 3):
            a, da = multiscale_crop(big, index, epoch)
            b, db = multiscale_crop(big, index, epoch)
            assert da == db
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_crop_carries_no_hidden_state(big):
    """Interleaving other draws must not shift the sequence. A generator held on
    the module would pass a naive twice-in-a-row check and fail this."""
    first, _ = multiscale_crop(big, 11, 2)
    for other in range(20):
        multiscale_crop(big, other, other)
    again, _ = multiscale_crop(big, 11, 2)
    assert np.array_equal(np.asarray(first), np.asarray(again))


def test_epochs_differ(big):
    """Otherwise every epoch of the cache is the same window and the multi-scale
    augmentation does nothing at all."""
    draws = {multiscale_crop(big, 11, e)[1] for e in range(12)}
    assert len(draws) > 1


def test_images_differ_within_an_epoch(big):
    draws = {multiscale_crop(big, i, 0)[1] for i in range(50)}
    assert len(draws) > 1


def test_position_varies(big):
    """Size variety is not enough -- a crop always taken from the same corner
    sees a fixed region of every image."""
    spots = {multiscale_crop(big, 5, e)[1].left for e in range(30)}
    assert len(spots) > 1


def test_seed_changes_the_draw(big):
    a = multiscale_crop(big, 3, 0, seed=0)[1]
    b = multiscale_crop(big, 3, 0, seed=1)[1]
    assert a != b


# --- geometry ----------------------------------------------------------------

def test_window_is_square_and_matches_the_draw(big):
    for epoch in range(8):
        img, draw = multiscale_crop(big, 17, epoch)
        assert img.size == (draw.size, draw.size)


def test_size_stays_in_range(big):
    for epoch in range(60):
        _, draw = multiscale_crop(big, 23, epoch, s_min=128, s_max=512)
        assert 128 <= draw.size <= 512


def test_window_never_leaves_the_source(big):
    w, h = big.size
    for epoch in range(60):
        _, d = multiscale_crop(big, 29, epoch)
        assert 0 <= d.left and d.left + d.size <= w
        assert 0 <= d.top and d.top + d.size <= h


def test_log_uniform_stays_in_range(big):
    for epoch in range(60):
        _, d = multiscale_crop(big, 31, epoch, s_min=128, s_max=512, policy="log_uniform")
        assert 128 <= d.size <= 512


def test_log_uniform_favours_finer_scales_than_uniform(big):
    """The reason the policy is a knob: drawing side length uniformly puts three
    quarters of the mass above 224, where preprocessing downsamples the crop and
    attenuates the traces it was taken to preserve."""
    u = np.mean([multiscale_crop(big, i, 0)[1].size for i in range(400)])
    lg = np.mean(
        [multiscale_crop(big, i, 0, policy="log_uniform")[1].size for i in range(400)]
    )
    assert lg < u


def test_unknown_policy_is_rejected(big):
    with pytest.raises(ValueError, match="policy"):
        multiscale_crop(big, 0, 0, policy="gaussian")


def test_bad_range_is_rejected(big):
    with pytest.raises(ValueError, match="s_min"):
        multiscale_crop(big, 0, 0, s_min=512, s_max=128)


# --- clamping: the leak this makes visible -----------------------------------

def test_a_small_source_clamps_and_says_so():
    """A 200x200 real cannot supply a 400px window. Clamping records the
    shortfall; upscaling would hide it, and the frequency branch would then read
    the interpolation instead of the label."""
    small = image(200, 200)
    draws = []
    for epoch in range(40):
        img, d = multiscale_crop(small, 1, epoch, s_min=128, s_max=512)
        assert img.size == (d.size, d.size)
        assert d.size == min(d.drawn, 200)
        assert d.clamped == (d.drawn > 200)
        draws.append(d)
    assert any(d.clamped for d in draws), "128-512 against a 200px source must clamp"
    assert any(not d.clamped for d in draws), "and must not always clamp"


def test_clamping_never_fires_when_the_range_fits():
    """The audit's whole job is to pick an s_max at which this holds for every
    source in the corpus, because a clamp rate that differs by class *is* the
    label leaking back in through crop size."""
    small = image(200, 200)
    for epoch in range(40):
        _, d = multiscale_crop(small, 1, epoch, s_min=128, s_max=200)
        assert not d.clamped
        assert d.size == d.drawn


def test_scale_reports_the_shortfall():
    small = image(160, 160)
    _, d = multiscale_crop(small, 2, 0, s_min=400, s_max=400)
    assert d.size == 160 and d.drawn == 400
    assert d.scale == pytest.approx(0.4)


def test_a_non_square_source_clamps_to_its_short_side():
    wide = image(1024, 300)
    for epoch in range(30):
        _, d = multiscale_crop(wide, 3, epoch, s_min=128, s_max=512)
        assert d.size <= 300


# --- the evaluation arms: deterministic, index-free --------------------------

def test_fixed_crop_is_deterministic_and_index_free(big):
    a, b = fixed_crop(big, 200), fixed_crop(big, 200)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert a.size == (200, 200)


def test_fixed_crop_at_200_is_a_no_op_on_a_200px_real():
    """Why arm (a) sits at exactly 200: it is the largest window every image in
    the corpus supplies from its own pixels, so the reals are passed through
    untouched and nothing is upsampled."""
    real = image(200, 200)
    assert np.array_equal(np.asarray(fixed_crop(real, 200)), np.asarray(real))


def test_fixed_crop_upscales_a_source_that_is_too_small():
    tiny = image(64, 64)
    assert fixed_crop(tiny, 200).size == (200, 200)


def test_fixed_crop_preserves_aspect_before_cropping():
    wide = image(400, 100)
    out = fixed_crop(wide, 200)
    assert out.size == (200, 200)


def test_fixed_resample_squares_everything(big):
    assert fixed_resample(big, 512).size == (512, 512)
    assert fixed_resample(image(1792, 1024), 512).size == (512, 512)
    assert fixed_resample(image(200, 200), 512).size == (512, 512)


def test_eval_arms_give_every_image_identical_dimensions():
    """The property the two-arm benchmark rests on: within an arm, dimensions
    carry no information, so the shortcut baseline is 0.5 by construction rather
    than by normalisation."""
    sources = [image(200, 200), image(1024, 1024), image(1792, 1024), image(346, 346)]
    assert {fixed_crop(s, 200).size for s in sources} == {(200, 200)}
    assert {fixed_resample(s, 512).size for s in sources} == {(512, 512)}


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_is_stable_and_parameter_sensitive():
    base = dict(s_min=128, s_max=512, seed=0, policy="uniform")
    assert crop_fingerprint(**base) == crop_fingerprint(**base)
    assert crop_fingerprint(**{**base, "s_max": 448}) != crop_fingerprint(**base)
    assert crop_fingerprint(**{**base, "policy": "log_uniform"}) != crop_fingerprint(**base)
    assert crop_fingerprint(**base, enabled=False) != crop_fingerprint(**base)


def test_a_whole_image_cache_cannot_pass_for_a_cropped_one():
    """The two protocols produce features of different windows of different
    images. Mixing them silently is the failure this hash exists to stop."""
    cropped = crop_fingerprint(128, 512, 0, "uniform", enabled=True)
    whole = crop_fingerprint(128, 512, 0, "uniform", enabled=False)
    assert cropped != whole


def test_crop_draw_is_hashable_and_comparable():
    """Drawn values go into `recipes.parquet` and into set-based assertions."""
    a = CropDraw(200, 0, 0, 200, False)
    assert a == CropDraw(200, 0, 0, 200, False)
    assert len({a, CropDraw(200, 0, 0, 200, False)}) == 1
