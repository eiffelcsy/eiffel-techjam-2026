"""The two evaluation arms, and the guard that decides which head may run them.

Training and evaluation deliberately use different protocols here: the head is
fit on random 128-512px windows, then scored on a fixed 200px window (arm a) and
on a whole-image 512 squash (arm b). `_assert_head_matches` used to require the
protocols be *equal*, which is right when a mismatch means nonsense and wrong
when the mismatch is the experiment. These tests pin the new boundary -- that
the permission is granted to exactly one training protocol and to nothing else.

No backbone is loaded: the guard takes a payload dict and the preprocess builder
takes a processor, so both are exercised against stubs.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from grace.cache.spec import sha_preprocess
from eval.detectors.dinov3 import (
    HEAD_COMPATIBILITY, INPUT_MODES, VIEWS, _assert_head_matches, _build_preprocess,
)
from eval.detectors.hf import (
    _CropPreprocess, _CropResizePreprocess, _ProcessorPreprocess, _ResamplePreprocess,
)


class StubProcessor:
    """Stands in for an `AutoImageProcessor`: records what it was handed, then
    squashes to 224 the way the real one does (`default_to_square: true`)."""

    def __init__(self):
        self.seen: list[tuple[int, int]] = []

    def __call__(self, img, return_tensors=None, do_resize=True):
        self.seen.append(img.size)
        out = img.resize((224, 224)) if do_resize else img
        arr = np.asarray(out, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return type("_P", (), {"pixel_values": torch.from_numpy(arr)[None]})()


def image(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mode="RGB")


def head(mode: str | None) -> dict:
    return {} if mode is None else {"input_mode": mode}


# --- the guard ---------------------------------------------------------------

def check(stored: str | None, feeding: str):
    _assert_head_matches(head(stored), "head.pt", "b", "cls", 384, feeding)


def test_a_head_runs_under_its_own_protocol():
    for mode in ("resize", "crop", "multiscale"):
        check(mode, mode)


def test_a_resize_head_still_refuses_crops():
    """The original guarantee. A resize-trained head handed native crops loads,
    runs, and scores nonsense -- silently, which is why this is an exception."""
    with pytest.raises(ValueError, match="input_mode"):
        check("resize", "crop")


def test_a_resize_head_refuses_the_eval_arms():
    for arm in ("crop200", "resample512"):
        with pytest.raises(ValueError, match="input_mode"):
            check("resize", arm)


def test_a_crop_head_refuses_everything_else():
    for other in ("resize", "multiscale", "crop200", "resample512"):
        with pytest.raises(ValueError, match="input_mode"):
            check("crop", other)


def test_a_multiscale_head_runs_both_eval_arms():
    """The permission the two-arm benchmark needs, and the only one granted."""
    check("multiscale", "crop200")
    check("multiscale", "resample512")


def test_a_multiscale_head_refuses_whole_image_resize():
    """`resample512` is a 512 squash the model can at least meet halfway; plain
    `resize` is the whole-image protocol multi-scale training exists to reject."""
    with pytest.raises(ValueError, match="input_mode"):
        check("multiscale", "resize")


def test_a_head_with_no_recorded_mode_is_treated_as_resize():
    """Heads predating the flag can only have been fit on the processor's own
    transform, so defaulting is what makes them refuse a crop rather than
    silently accept one."""
    check(None, "resize")
    with pytest.raises(ValueError, match="input_mode"):
        check(None, "crop200")


def test_the_error_names_what_the_head_could_run():
    with pytest.raises(ValueError, match="crop200"):
        check("multiscale", "resize")


def test_an_unknown_stored_mode_falls_back_to_equality():
    """A checkpoint from a protocol this build has never heard of must not be
    silently permitted; the safe reading is that it runs only under itself."""
    check("someday_mode", "someday_mode")
    with pytest.raises(ValueError, match="input_mode"):
        check("someday_mode", "crop200")


# --- table consistency -------------------------------------------------------

def test_every_compatibility_entry_names_real_modes():
    for stored, allowed in HEAD_COMPATIBILITY.items():
        assert stored in INPUT_MODES
        assert allowed <= set(INPUT_MODES)


def test_every_view_is_a_declared_mode():
    assert set(VIEWS) <= set(INPUT_MODES)


def test_every_mode_builds_a_preprocess():
    for mode in INPUT_MODES:
        assert _build_preprocess(StubProcessor(), mode, 224) is not None


# --- the transforms the arms actually apply ----------------------------------

def test_modes_resolve_to_the_expected_transform():
    p = StubProcessor()
    assert isinstance(_build_preprocess(p, "resize", 224), _ProcessorPreprocess)
    assert isinstance(_build_preprocess(p, "crop", 224), _CropPreprocess)
    assert isinstance(_build_preprocess(p, "crop200", 224), _CropResizePreprocess)
    assert isinstance(_build_preprocess(p, "resample512", 224), _ResamplePreprocess)


def test_multiscale_is_the_plain_processor_transform():
    """The window was already drawn in the dataset, where it could be seeded on
    the image index. If this became a cropping transform it would be stochastic
    and `sha_preprocess` would refuse the cache."""
    assert isinstance(_build_preprocess(StubProcessor(), "multiscale", 224),
                      _ProcessorPreprocess)


def test_arm_a_crops_to_200_before_the_model_sees_anything():
    p = StubProcessor()
    fn = _build_preprocess(p, "crop200", 224)
    for src in [(200, 200), (1024, 1024), (1792, 1024), (346, 346)]:
        fn(image(*src))
    assert p.seen == [(200, 200)] * 4


def test_arm_b_resamples_to_512_before_the_model_sees_anything():
    p = StubProcessor()
    fn = _build_preprocess(p, "resample512", 224)
    for src in [(200, 200), (1024, 1024), (1792, 1024)]:
        fn(image(*src))
    assert p.seen == [(512, 512)] * 3


def test_both_arms_erase_the_dimension_shortcut():
    """The property the repaired benchmark rests on: within an arm, every image
    has identical dimensions, so `max(w, h)` carries no information at all. On
    the raw benchmark that same statistic scores AUC 1.0000."""
    sources = [image(200, 200), image(1024, 1024), image(1792, 1024), image(346, 346)]
    for mode in ("crop200", "resample512"):
        p = StubProcessor()
        fn = _build_preprocess(p, mode, 224)
        for src in sources:
            fn(src)
        assert len(set(p.seen)) == 1


def test_every_arm_outputs_the_model_input_size():
    for mode in INPUT_MODES:
        fn = _build_preprocess(StubProcessor(), mode, 224)
        assert fn(image(1024, 1024)).shape == (3, 224, 224)


# --- cacheability ------------------------------------------------------------

@pytest.mark.parametrize("mode", INPUT_MODES)
def test_every_mode_is_cacheable(mode):
    """`sha_preprocess` runs the transform twice and refuses a stochastic one.
    Every eval arm has to pass it, or the arm cannot have a rendered cache --
    which is the whole reason the multi-scale draw lives in the dataset instead
    of in here."""
    fn = _build_preprocess(StubProcessor(), mode, 224)
    assert sha_preprocess(fn) == sha_preprocess(fn)


# --- the mode and the crop config must not disagree --------------------------

def probe_cfg(**kw):
    from grace.config import CropConfig

    return type("_C", (), {"crop": CropConfig(**kw)})()


def split_with(mode: str):
    return type("_S", (), {"detector": type("_D", (), {"input_mode": mode})()})()


def test_crops_without_the_mode_are_refused():
    """The head records `input_mode` and nothing else about the window. Fitting
    on crops while the config still said `resize` would stamp the head `resize`,
    let it past the evaluation guard against a whole-image detector, and score a
    feature space it never saw -- the exact silent failure the guard exists for,
    reintroduced from the training side."""
    from grace.probe.train import _assert_crop_matches_input_mode

    cfg = probe_cfg(enabled=True, s_max=200)
    with pytest.raises(ValueError, match="input_mode"):
        _assert_crop_matches_input_mode(cfg, split_with("resize"))


def test_the_mode_without_crops_is_refused():
    """The other direction: a head named for windows that were never drawn."""
    from grace.probe.train import _assert_crop_matches_input_mode

    with pytest.raises(ValueError, match="crop.enabled"):
        _assert_crop_matches_input_mode(probe_cfg(), split_with("multiscale"))


def test_the_agreeing_combinations_pass():
    from grace.probe.train import _assert_crop_matches_input_mode

    _assert_crop_matches_input_mode(probe_cfg(), split_with("resize"))
    _assert_crop_matches_input_mode(probe_cfg(), split_with("crop"))
    _assert_crop_matches_input_mode(
        probe_cfg(enabled=True, s_max=200), split_with("multiscale")
    )


def test_the_arms_are_distinguishable_by_fingerprint():
    """Two arms that hashed alike could silently share a cache, and their
    features are of different images."""
    a = sha_preprocess(_build_preprocess(StubProcessor(), "crop200", 224))
    b = sha_preprocess(_build_preprocess(StubProcessor(), "resample512", 224))
    assert a != b
