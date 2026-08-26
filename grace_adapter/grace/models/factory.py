"""FeatureSpec -> modules. The only place layout is branched on.

The branch is three lines, and it decides a *gate shape*, not a class. Sharing
the MLP across the group axis and varying only the gate is what keeps parameter
count flat in `L` and `T`, and it is why there is one adapter class rather than
three.
"""

import torch
import torch.nn as nn

from grace.models.adapter import GatedResidualAdapter
from grace.models.discrepancy import DiscrepancyHead
from grace.models.severity import SeverityHead
from grace.splits.base import FeatureSpec


def gate_shape_for(spec: FeatureSpec, per_channel_gate: bool = True) -> tuple[int, ...]:
    """`layers` gets one gate vector per layer; everything else one per channel.

    The per-layer gate is the point of the RINE figure: early CLIP blocks carry
    the high-frequency generation traces that blur destroys, late ones carry
    semantics that survive it, and a shared gate cannot say so.
    """
    if not per_channel_gate:
        return ()
    if spec.layout == "layers":
        return (spec.n_groups, spec.dim)
    return (spec.dim,)


def build_adapter(spec: FeatureSpec, cfg) -> GatedResidualAdapter:
    return GatedResidualAdapter(
        dim=spec.dim,
        gate_shape=gate_shape_for(spec, cfg.per_channel_gate),
        bottleneck=cfg.bottleneck,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        noise_dim=cfg.noise_dim,
        severity_film=cfg.severity_film,
    )


def build_severity_head(spec: FeatureSpec, hidden: int = 256) -> SeverityHead:
    return SeverityHead(dim=spec.dim, hidden=hidden)


def build_discrepancy_head(spec: FeatureSpec, cfg) -> DiscrepancyHead:
    return DiscrepancyHead(
        spec=spec, hidden=cfg.hidden, proj=cfg.proj, use_severity=cfg.use_severity
    )


def save_adapter(path, adapter: nn.Module, spec: FeatureSpec, cfg, extra: dict | None = None):
    """Weights *and* the config that shaped them.

    An adapter checkpoint must be loadable by the eval harness with no reference
    to the training run that produced it -- that is what lets
    `configs/detectors/*+grace.yaml` name a checkpoint and nothing else.
    """
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "feature_spec": spec.to_dict(),
            "adapter_cfg": vars(cfg),
            **(extra or {}),
        },
        path,
    )


def load_adapter(checkpoint: str, spec: FeatureSpec | None = None) -> GatedResidualAdapter:
    """Rebuild from the checkpoint's stored config and load its weights.

    If `spec` is given it is checked against the stored one: loading an adapter
    trained on a different feature layout would otherwise fail deep inside a
    matmul rather than here.
    """
    from grace.config import AdapterConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = FeatureSpec.from_dict(payload["feature_spec"])
    if spec is not None and (spec.layout, spec.shape) != (stored.layout, stored.shape):
        raise ValueError(
            f"{checkpoint} was trained on {stored.layout}{stored.shape} but this "
            f"detector emits {spec.layout}{spec.shape}"
        )
    adapter = build_adapter(stored, AdapterConfig(**payload["adapter_cfg"]))
    adapter.load_state_dict(payload["state_dict"])
    return adapter.eval()
