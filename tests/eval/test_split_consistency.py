"""head(trunk(x)) == detector(x). Break this and every comparison is against a
model that was never benchmarked.

The zoo splits themselves need their vendored repos and cannot run here, which is
exactly why `verify_split` exists and is called from every split's `__init__`:
the check runs on the machine that has the weights, at construction, before a
single feature is cached. These tests pin the checker.
"""

import pytest
import torch
import torch.nn as nn

from eval.splits import build_split
from eval.splits.base import FeatureSpec
from eval.splits.verify import verify_split
from tests.fixtures import SPECS, MLPHead, ToySplit


@pytest.mark.parametrize("layout", list(SPECS))
def test_split_reproduces_forward(layout):
    split = ToySplit(SPECS[layout])
    x = torch.randn(4, 8)
    assert torch.allclose(split.head(split.trunk(x)), split.detector(x), atol=1e-5)


@pytest.mark.parametrize("layout", list(SPECS))
def test_feature_spec_matches_trunk_output(layout):
    """The cache writer commits to the declared spec before the first batch."""
    spec = SPECS[layout]
    split = ToySplit(spec)
    assert split.trunk(torch.randn(3, 8)).shape == (3, *spec.shape)


@pytest.mark.parametrize("head", [None, "mlp"])
def test_verify_accepts_a_correct_split(head):
    spec = SPECS["vector"]
    verify_split(ToySplit(spec, head=MLPHead(spec) if head else None))


def test_verify_rejects_a_wrong_head():
    """A composition error must be loud. This is the failure that would otherwise
    produce plausible logits from the wrong model for a whole week."""
    split = ToySplit(SPECS["vector"])
    split.head = lambda f: torch.zeros(f.shape[0])
    with pytest.raises(RuntimeError, match="max .difference."):
        verify_split(split)


def test_verify_rejects_a_head_returning_none():
    split = ToySplit(SPECS["vector"])
    split.head = lambda f: None
    with pytest.raises(RuntimeError, match="returned None"):
        verify_split(split)


def test_verify_message_lists_trainable_modules():
    """So that fixing `_head_forward` does not mean reading the clone from
    scratch."""
    split = ToySplit(SPECS["vector"])
    split.head = lambda f: torch.zeros(f.shape[0])
    split.head_modules = lambda: {"proj": nn.Linear(2, 2)}
    with pytest.raises(RuntimeError, match="proj: Linear"):
        verify_split(split)


def test_frozen_split_has_no_trainable_parameters():
    split = ToySplit(SPECS["vector"])
    split.assert_frozen()


def test_assert_frozen_catches_train_mode():
    """Checked every step, not once: a BatchNorm detector left in train mode
    adapts itself on degraded data and contaminates the comparison."""
    split = ToySplit(SPECS["vector"])
    split.detector.train()
    with pytest.raises(RuntimeError, match="training mode"):
        split.assert_frozen()


def test_assert_frozen_catches_trainable_parameters():
    split = ToySplit(SPECS["vector"])
    split.detector.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable parameter"):
        split.assert_frozen()


def test_build_split_rejects_a_non_split():
    with pytest.raises(TypeError, match="not a SplitDetector"):
        build_split(nn.Linear(2, 2), "torch.nn.Identity")


def test_feature_spec_validates_rank():
    with pytest.raises(ValueError, match="2-d shape"):
        FeatureSpec(layout="layers", shape=(16,))
    with pytest.raises(ValueError, match="layout must be"):
        FeatureSpec(layout="nonsense", shape=(16,))


def test_verify_uses_an_image_shaped_probe():
    """It builds its own input from the detector's declared `input_size`, so a
    split can be checked with no dataset present."""
    from eval.splits.verify import _probe_shape

    split = ToySplit(SPECS["vector"])
    assert _probe_shape(split) == (3, 224, 224)
    split.detector.input_size = 96
    assert _probe_shape(split) == (3, 96, 96)
