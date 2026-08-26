"""The discrepancy branch: fusion must start as a no-op, and Δ must be readable.

The β=0 initialization is what makes GRACE-D and GRACE comparable at all. If
wiring in the auxiliary head changed the reported numbers on its own, no
difference between the two variants would be attributable to what the head
learned.
"""

import pytest
import torch

from grace.config import AdapterConfig, DiscrepancyConfig
from grace.models.discrepancy import FusedHead
from grace.models.factory import build_adapter, build_discrepancy_head
from grace.train import diagnostics as D
from tests.fixtures import SPECS, LinearHead, features


@pytest.mark.parametrize("layout", list(SPECS))
def test_fusion_is_identity_at_init(layout):
    spec = SPECS[layout]
    fused = FusedHead(build_discrepancy_head(spec, DiscrepancyConfig()))
    delta = features(spec)
    main = torch.randn(delta.shape[0])
    sev = torch.rand(delta.shape[0])
    assert torch.allclose(fused(main, delta, sev), main, atol=1e-7)


@pytest.mark.parametrize("layout", list(SPECS))
def test_head_is_layout_agnostic(layout):
    spec = SPECS[layout]
    head = build_discrepancy_head(spec, DiscrepancyConfig())
    delta = features(spec)
    assert head(delta, torch.rand(delta.shape[0])).shape == (delta.shape[0],)


def test_per_group_norms_are_the_damage_profile():
    """For `layers` the head's first inputs are one norm per block -- the same
    vector the per-layer gate produces, which is why it doubles as the figure."""
    spec = SPECS["layers"]
    head = build_discrepancy_head(spec, DiscrepancyConfig(use_severity=False))
    delta = features(spec)
    feats = head.features(delta)
    assert torch.allclose(feats[:, : spec.n_groups], torch.log1p(delta.norm(dim=-1)))


def test_severity_is_required_when_declared():
    head = build_discrepancy_head(SPECS["vector"], DiscrepancyConfig(use_severity=True))
    with pytest.raises(ValueError, match="use_severity"):
        head(features(SPECS["vector"]), None)


def test_delta_from_an_untrained_adapter_is_zero():
    """Δ is identically zero at init, so an untrained adapter carries no drift
    estimate at all -- stage 2 is meaningless before stage 1 has run, and this is
    the assertion that says so."""
    spec = SPECS["vector"]
    adapter = build_adapter(spec, AdapterConfig())
    f = features(spec)
    assert torch.allclose(adapter(f) - f, torch.zeros_like(f), atol=1e-7)


def test_drift_asymmetry_detects_a_planted_gap():
    """The premise check must actually fire when generated features drift more."""
    spec = SPECS["vector"]
    clean = features(spec, batch=256)
    labels = torch.zeros(256, dtype=torch.long)
    labels[128:] = 1
    noise = torch.randn_like(clean) * 0.01
    noise[128:] *= 5.0                      # fakes drift five times as far
    stats = D.drift_asymmetry(clean + noise, clean, labels)
    assert stats["asymmetry"] > 0
    assert stats["drift_fake"] > stats["drift_real"]


def test_drift_asymmetry_reports_the_decision_subspace_split():
    spec = SPECS["vector"]
    clean = features(spec, batch=64)
    labels = torch.randint(0, 2, (64,))
    j = LinearHead(spec).w.detach().reshape(1, -1).expand_as(clean)
    stats = D.drift_asymmetry(clean + 0.1 * torch.randn_like(clean), clean, labels, j)
    assert 0.0 <= stats["parallel_fraction"] <= 1.0
    assert "orthogonal_asymmetry" in stats
