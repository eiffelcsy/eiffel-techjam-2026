"""The new objective terms, and the properties the ablations depend on."""

import pytest
import torch
import torch.nn.functional as F

from grace.train.losses import alignment_loss, head_kl, identity_loss, sliced_wasserstein
from grace.train.weighting import decision_weighted_error, head_gradient
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


@pytest.mark.parametrize("layout", list(SPECS))
def test_sliced_wasserstein_is_zero_for_identical_batches(layout):
    spec = SPECS[layout]
    a = features(spec, batch=64)
    assert float(sliced_wasserstein(a, a.clone(), n_proj=32)) == pytest.approx(0.0, abs=1e-6)


def test_sliced_wasserstein_is_permutation_invariant():
    """It compares distributions, not pairs -- unlike every other term here."""
    spec = SPECS["vector"]
    a = features(spec, batch=64)
    torch.manual_seed(0)
    shuffled = a[torch.randperm(a.shape[0])]
    torch.manual_seed(0)
    same = sliced_wasserstein(a, shuffled, n_proj=32)
    assert float(same) == pytest.approx(0.0, abs=1e-6)


def test_sliced_wasserstein_penalises_under_dispersion():
    """The property that gives the adapter's noise input a job: a collapsed,
    over-concentrated batch is penalised even when its mean is right."""
    spec = SPECS["vector"]
    clean = features(spec, batch=128)
    collapsed = clean.mean(0, keepdim=True).expand_as(clean).contiguous()
    torch.manual_seed(0)
    a = float(sliced_wasserstein(collapsed, clean, n_proj=64))
    torch.manual_seed(0)
    b = float(sliced_wasserstein(clean * 1.0, clean, n_proj=64))
    assert a > b


@pytest.mark.parametrize("layout", list(SPECS))
def test_identity_loss_is_zero_at_init(layout):
    from grace.config import AdapterConfig
    from grace.models.factory import build_adapter

    spec = SPECS[layout]
    adapter = build_adapter(spec, AdapterConfig())
    f = features(spec)
    assert float(identity_loss(adapter, f).detach()) == pytest.approx(0.0, abs=1e-10)


def test_head_kl_is_zero_when_features_match():
    spec = SPECS["vector"]
    head, f = MLPHead(spec), features(spec)
    assert float(head_kl(head, f, f)) == pytest.approx(
        float(head_kl(head, f, f)), abs=1e-9
    )
    assert float(head_kl(head, f, f)) >= 0.0
