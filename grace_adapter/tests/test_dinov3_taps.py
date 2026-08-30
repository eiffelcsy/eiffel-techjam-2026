"""What only a real ViT can check: that the tapped forward IS the seam forward.

`tests/test_ladder.py` covers the ladder's plumbing on the toy split. The
properties here are arithmetic facts about DINOv3 specifically, and they are the
ones that make the ladder's numbers comparable to the plain adapter's:

  * `trunk_with_taps(x)[0]` reproduces `trunk(x)` bit-for-bit, so a ladder arm is
    scored at the seam every baseline was measured at.
  * a tap at the LAST hidden state pools to exactly the seam feature, because
    `last_hidden_state == backbone.norm(hidden_states[-1])`. That equality is
    what justifies pushing every tap through `backbone.norm`, and if a
    transformers release changes it this test says so.

Both run against the tiny randomly-initialized DINOv3 the PoC tests already
build, so no gated weights and no network are involved.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.dinov3_vit")

from eval.splits.dinov3 import DEFAULT_TAP_BLOCKS, DINOv3Split      # noqa: E402
from eval.splits.verify import verify_taps                          # noqa: E402
from tests.test_dinov3_poc import (                                  # noqa: E402,F401
    HIDDEN, IMAGE, detector_factory, tiny_backbone,
)

# The tiny backbone is 2 blocks deep, so hidden_states is (embedding, b1, b2).
N_STATES = 3
TAPS = (0, 1)


def _split(detector_factory, **kw):
    with pytest.warns(UserWarning, match="randomly initialized head"):
        split = DINOv3Split(detector_factory(**kw), **kw.pop("split_kw", {}))
    return split.eval()


def _tapped(detector_factory, tap_blocks, **kw):
    with pytest.warns(UserWarning, match="randomly initialized head"):
        split = DINOv3Split(detector_factory(**kw), tap_blocks=tap_blocks)
    return split.eval()


def test_no_tap_blocks_leaves_the_split_exactly_as_it_was(detector_factory):
    """The default has to be inert: every existing cache, config and checkpoint
    was made by a split that emitted nothing."""
    split = _tapped(detector_factory, None)
    assert split.taps() == ()
    assert split.tap_spec() is None
    x = torch.randn(2, 3, IMAGE, IMAGE)
    with torch.no_grad():
        f, taps = split.trunk_with_taps(x)
    assert taps is None
    assert torch.equal(f, split.trunk(x))


def test_tapped_forward_reproduces_the_seam_exactly(detector_factory):
    """Not `allclose` -- equal. A ladder scored at a seam that drifts from the
    baseline's by even a float epsilon is measuring a different model."""
    split = _tapped(detector_factory, TAPS)
    x = torch.randn(3, 3, IMAGE, IMAGE)
    with torch.no_grad():
        f, taps = split.trunk_with_taps(x)
    assert torch.equal(f, split.trunk(x))
    assert taps.shape == (3, len(TAPS), HIDDEN * 2)


def test_a_tap_at_the_last_block_is_the_seam_feature(detector_factory):
    """`last_hidden_state == norm(hidden_states[-1])`, which is why every tap
    goes through `backbone.norm`. This is the free exactness check the shared
    `pool_tokens` buys, and the canary if transformers moves the final norm."""
    split = _tapped(detector_factory, (N_STATES - 1,))
    x = torch.randn(2, 3, IMAGE, IMAGE)
    with torch.no_grad():
        f, taps = split.trunk_with_taps(x)
    assert torch.equal(taps[:, 0], f)


def test_taps_are_named_by_the_block_they_read(detector_factory):
    split = _tapped(detector_factory, (0, 2))
    assert split.taps() == ("block00", "block02")
    assert split.tap_spec().shape == (2, HIDDEN * 2)


@pytest.mark.parametrize("pool,mult", [("cls", 1), ("patchmean", 1), ("cls+patchmean", 2)])
def test_taps_are_pooled_exactly_as_the_seam_is(detector_factory, pool, mult):
    """A tap pooled differently from the seam would hand the ladder a feature
    space the rest of the adapter has never seen."""
    split = _tapped(detector_factory, TAPS, pool=pool)
    assert split.tap_spec().shape == (len(TAPS), HIDDEN * mult)
    assert split.tap_spec().dim == split.feature_spec.dim


def test_out_of_range_tap_blocks_are_refused(detector_factory):
    with pytest.raises(ValueError, match="out of range"):
        _tapped(detector_factory, (0, N_STATES))
    with pytest.raises(ValueError, match="duplicates"):
        _tapped(detector_factory, (1, 1))


def test_verify_taps_runs_at_construction(detector_factory):
    """Construction already ran it; this pins that it is wired, since a split
    that stopped verifying would fail nowhere until results were wrong."""
    verify_taps(_tapped(detector_factory, TAPS))


def test_default_tap_blocks_fit_a_twelve_block_vit():
    """The documented set is indices into `hidden_states` (0 = embedding), and
    excludes the last block because that one IS the seam."""
    assert max(DEFAULT_TAP_BLOCKS) < 12
    assert len(set(DEFAULT_TAP_BLOCKS)) == len(DEFAULT_TAP_BLOCKS)
