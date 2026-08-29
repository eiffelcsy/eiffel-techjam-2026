"""At initialization the adapter must be exactly the identity, not approximately.

Approximate is not good enough: an adapter that perturbs clean features at step 0
makes any clean-AUC change unattributable. The guarantee has to survive every
optional input -- severity conditioning, dropout -- because each of them is a
place a future change could break it silently.
"""

import pytest
import torch

from grace.config import AdapterConfig
from grace.models.adapter import GATE_INIT, GatedResidualAdapter
from grace.models.factory import build_adapter, gate_shape_for
from tests.fixtures import SPECS, features


@pytest.mark.parametrize("layout", list(SPECS))
def test_identity_at_init(layout):
    spec = SPECS[layout]
    adapter = build_adapter(spec, AdapterConfig())
    f = features(spec)
    assert torch.allclose(adapter(f), f, atol=1e-6)


@pytest.mark.parametrize("layout", list(SPECS))
def test_identity_holds_for_any_severity(layout):
    """The zero-init of the last projection makes the correction identically zero
    whatever conditions it -- which is why the optional input is safe to add."""
    spec = SPECS[layout]
    adapter = build_adapter(
        spec, AdapterConfig(severity_film=True, dropout=0.3)
    )
    f = features(spec)
    sev = torch.rand(f.shape[0])
    assert torch.allclose(adapter(f, severity=sev), f, atol=1e-6)


def test_gate_starts_low():
    adapter = GatedResidualAdapter(dim=16)
    assert torch.allclose(adapter.gate(), torch.sigmoid(torch.tensor(GATE_INIT)), atol=1e-6)
    assert float(adapter.gate().mean().detach()) < 0.02


@pytest.mark.parametrize("layout", list(SPECS))
def test_identity_survives_a_swept_gate_init(layout):
    """`gate_init` is a sweepable config key, and sweeping it must not quietly
    cost the guarantee the whole comparison rests on. `fc2` is zero-init, so the
    correction is identically zero for ANY gate -- this is what makes a
    gate_init arm attributable against its control."""
    spec = SPECS[layout]
    cfg = AdapterConfig(gate_init=-3.0, severity_film=True)
    adapter = build_adapter(spec, cfg)
    f = features(spec)

    assert torch.allclose(adapter.gate(), torch.sigmoid(torch.tensor(-3.0)), atol=1e-6)
    assert torch.allclose(adapter(f), f, atol=1e-6)
    assert torch.allclose(adapter(f, severity=torch.rand(f.shape[0])), f, atol=1e-6)


@pytest.mark.parametrize("layout", list(SPECS))
def test_layout_roundtrips_shape(layout):
    spec = SPECS[layout]
    adapter = build_adapter(spec, AdapterConfig())
    f = features(spec)
    assert adapter(f).shape == f.shape


def test_layers_gate_is_per_layer():
    """The `layers` gate is (L, D): the per-block damage profile, and the figure."""
    spec = SPECS["layers"]
    assert gate_shape_for(spec) == (spec.n_groups, spec.dim)
    adapter = build_adapter(spec, AdapterConfig())
    assert adapter.gate().shape == (spec.n_groups, spec.dim)


def test_tokens_gate_is_shared_across_tokens():
    """Corruption is a property of the image, not of token position."""
    assert gate_shape_for(SPECS["tokens"]) == (SPECS["tokens"].dim,)


@pytest.mark.parametrize("layout", list(SPECS))
def test_mlp_is_shared_across_the_group_axis(layout):
    """Parameter count must not grow with L or T -- that is what keeps the
    adapter tiny for RINE's 24 layers and B-Free's 5 windows."""
    spec = SPECS[layout]
    n = sum(p.numel() for p in build_adapter(spec, AdapterConfig()).fc1.parameters())
    assert n == 2 * (spec.dim * 256 + 256)      # n_blocks=2, bottleneck=256


def test_conditioning_changes_the_gate_once_trained():
    """FiLM starts as a no-op; give it nonzero weights and severity must matter."""
    adapter = GatedResidualAdapter(dim=16, severity_film=True)
    torch.nn.init.normal_(adapter.film.weight, std=0.5)
    low = adapter.gate(torch.zeros(4))
    high = adapter.gate(torch.ones(4))
    assert not torch.allclose(low, high)
