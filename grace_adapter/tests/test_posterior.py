"""Posterior sampling: it must be optional, exact when off, and honest when on.

The load-bearing claim is that averaging *logits* over k draws is not the same as
one pass through the mean correction, because the head is nonlinear. If that were
false the whole feature would be a waste of k forward passes.
"""

import pytest
import torch

from grace.config import AdapterConfig
from grace.models.factory import build_adapter
from grace.train import diagnostics as D
from tests.fixtures import SPECS, LinearHead, MLPHead, features


def _trained(spec, **kw):
    """An adapter perturbed off its zero-init, so corrections are nonzero."""
    adapter = build_adapter(spec, AdapterConfig(**kw))
    with torch.no_grad():
        for layer in adapter.fc2:
            torch.nn.init.normal_(layer.weight, std=0.05)
        adapter.gate_logit.fill_(0.0)
    return adapter


@pytest.mark.parametrize("layout", list(SPECS))
def test_deterministic_adapter_is_deterministic(layout):
    spec = SPECS[layout]
    adapter = _trained(spec)
    f = features(spec)
    assert not adapter.stochastic
    draws = adapter.sample(f, k=4)
    assert torch.allclose(draws[0], draws[-1], atol=1e-7)


def test_passing_noise_to_a_deterministic_adapter_is_an_error():
    """Silently ignoring `z` would make a misconfigured sweep look like a
    negative result about posterior sampling."""
    spec = SPECS["vector"]
    adapter = build_adapter(spec, AdapterConfig(noise_dim=0))
    with pytest.raises(ValueError, match="noise_dim=0"):
        adapter(features(spec), z=torch.randn(8, 4))


@pytest.mark.parametrize("layout", list(SPECS))
def test_stochastic_adapter_actually_varies(layout):
    spec = SPECS[layout]
    adapter = _trained(spec, noise_dim=8)
    with torch.no_grad():
        for layer in adapter.noise:
            torch.nn.init.normal_(layer.weight, std=0.5)
    draws = adapter.sample(features(spec), k=4)
    assert not torch.allclose(draws[0], draws[1], atol=1e-5)


def test_logit_averaging_differs_from_feature_averaging():
    """E[h(f)] != h(E[f]) -- the entire justification for k passes at eval."""
    spec = SPECS["vector"]
    head = MLPHead(spec)
    adapter = _trained(spec, noise_dim=8)
    with torch.no_grad():
        for layer in adapter.noise:
            torch.nn.init.normal_(layer.weight, std=1.0)

    torch.manual_seed(0)
    draws = adapter.sample(features(spec, batch=64), k=16)
    mean_of_logits = torch.stack([head(d) for d in draws]).mean(0)
    logit_of_mean = head(draws.mean(0))
    assert not torch.allclose(mean_of_logits, logit_of_mean, atol=1e-4)


def test_logit_averaging_matches_for_a_linear_head():
    """The converse, as a sanity check on the above: for a linear head the two
    are identical, so a detector with a purely linear head gains nothing here."""
    spec = SPECS["vector"]
    head = LinearHead(spec)
    adapter = _trained(spec, noise_dim=8)
    with torch.no_grad():
        for layer in adapter.noise:
            torch.nn.init.normal_(layer.weight, std=1.0)
    draws = adapter.sample(features(spec, batch=32), k=8)
    assert torch.allclose(
        torch.stack([head(d) for d in draws]).mean(0), head(draws.mean(0)), atol=1e-4
    )


def test_posterior_spread_flags_collapse():
    """The tripwire. A collapsed posterior reads ~0 and is a reportable negative
    result about the objective, not a bug to paper over."""
    collapsed = torch.zeros(8, 32) + torch.randn(1, 32)
    assert D.posterior_spread(collapsed) == pytest.approx(0.0, abs=1e-6)
    assert D.posterior_spread(torch.randn(8, 32)) > 0.1


def test_spread_of_a_single_draw_is_zero():
    assert D.posterior_spread(torch.randn(1, 32)) == 0.0
