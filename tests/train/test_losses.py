"""The new objective terms, and the properties the ablations depend on."""

import pytest
import torch
import torch.nn.functional as F

from train.losses import alignment_loss, head_kl, orthogonality_loss
from train.weighting import decision_weighted_error, head_gradient
from tests.fixtures import SPECS, LinearHead, MLPHead, features


@pytest.mark.parametrize("layout", list(SPECS))
def test_head_gradient_of_a_linear_head_is_exactly_w(layout):
    """The claim that makes one implementation cover both head kinds."""
    spec = SPECS[layout]
    head = LinearHead(spec)
    f = features(spec)
    j = head_gradient(head, f)
    expected = head.w.detach().reshape(1, *spec.shape).expand_as(f)
    assert torch.allclose(j, expected, atol=1e-6)


@pytest.mark.parametrize("layout", list(SPECS))
def test_head_gradient_is_per_sample(layout):
    """Summing the batch before backward must not mix samples together."""
    spec = SPECS[layout]
    head = MLPHead(spec)
    f = features(spec)
    j = head_gradient(head, f)
    alone = head_gradient(head, f[2:3])
    assert torch.allclose(j[2:3], alone, atol=1e-6)


def test_head_gradient_does_not_leak_into_the_caller_graph():
    spec = SPECS["vector"]
    f = features(spec).requires_grad_(True)
    assert not head_gradient(MLPHead(spec), f).requires_grad


@pytest.mark.parametrize("layout", list(SPECS))
def test_eps_one_is_exactly_plain_mse(layout):
    """The plain-MSE ablation must be provably the GRACE v1 objective, not an
    approximation of it -- otherwise the comparison measures two changes."""
    spec = SPECS[layout]
    a, b = features(spec, seed=1), features(spec, seed=2)
    j = head_gradient(LinearHead(spec), b)
    assert torch.allclose(
        decision_weighted_error(a - b, j, eps_iso=1.0), F.mse_loss(a, b), atol=1e-7
    )


def test_weighting_ignores_error_orthogonal_to_the_decision_direction():
    """The whole point: capacity spent orthogonal to `w` cannot move the logit."""
    spec = SPECS["vector"]
    head = LinearHead(spec)
    f = features(spec)
    j = head_gradient(head, f)

    w = F.normalize(head.w.detach(), dim=0)
    orthogonal = torch.randn(f.shape[0], spec.dim)
    orthogonal -= (orthogonal @ w).unsqueeze(1) * w          # project out
    parallel = torch.randn(f.shape[0], 1) * w

    # equal-energy errors, one visible to the head and one not
    orthogonal = F.normalize(orthogonal, dim=1)
    parallel = F.normalize(parallel, dim=1)
    lo = decision_weighted_error(orthogonal, j, eps_iso=0.0)
    hi = decision_weighted_error(parallel, j, eps_iso=0.0)
    assert lo < 1e-6 < hi


def test_alignment_weighting_none_matches_v1():
    spec = SPECS["vector"]
    a, b = features(spec, seed=1), features(spec, seed=2)
    manual = 1.0 * (1 - (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)).mean()
    manual = manual + (a - b).pow(2).mean()
    assert torch.allclose(alignment_loss(a, b, weighting="none"), manual, atol=1e-6)


def test_head_kl_is_zero_when_features_match():
    spec = SPECS["vector"]
    head, f = MLPHead(spec), features(spec)
    assert float(head_kl(head, f, f)) == pytest.approx(
        float(head_kl(head, f, f)), abs=1e-9
    )
    assert float(head_kl(head, f, f)) >= 0.0


def test_orthogonality_loss_is_zero_for_orthogonal_updates():
    """An update that points exactly orthogonal to the spatial feature carries
    the minimum possible redundancy -- the term must vanish."""
    spec = SPECS["vector"]
    spatial = torch.randn(8, spec.dim)
    s = F.normalize(spatial, dim=1)
    orth = torch.randn(8, spec.dim)
    orth = orth - (orth * s).sum(-1, keepdim=True) * s       # project out
    orth = F.normalize(orth, dim=1)
    assert float(orthogonality_loss(orth, spatial)) < 1e-6


def test_orthogonality_loss_is_maximal_for_parallel_updates():
    """A copy of the spatial direction is exactly the redundancy the term is
    there to price -- correlation one, so the penalty is one."""
    spec = SPECS["vector"]
    spatial = torch.randn(8, spec.dim)
    parallel = F.normalize(spatial, dim=1) * 3.0              # same direction
    assert float(orthogonality_loss(parallel, spatial)) == pytest.approx(1.0)

    # and an anti-parallel copy is just as redundant as a parallel one
    anti = -parallel
    assert float(orthogonality_loss(anti, spatial)) == pytest.approx(1.0)


def test_orthogonality_loss_ignores_update_magnitude():
    """The term must price DIRECTION, not size -- the gate owns magnitude, and a
    term that rewarded small updates would let the branch 'satisfy' it by
    shrinking instead of by learning."""
    spec = SPECS["vector"]
    spatial = torch.randn(8, spec.dim)
    s = F.normalize(spatial, dim=1)
    orth = torch.randn(8, spec.dim)
    orth = orth - (orth * s).sum(-1, keepdim=True) * s
    assert float(orthogonality_loss(orth * 1e-4, spatial)) < 1e-6
    assert float(orthogonality_loss(orth * 1e4, spatial)) < 1e-6
