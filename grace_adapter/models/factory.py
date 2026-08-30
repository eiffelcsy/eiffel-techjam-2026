"""FeatureSpec -> modules. The only place layout is branched on.

The branch is three lines, and it decides a *gate shape*, not a class. Sharing
the MLP across the group axis and varying only the gate is what keeps parameter
count flat in `L` and `T`, and it is why there is one adapter class rather than
three.

The enricher's build/save/load live in `freq_branch.models.factory` instead of
here -- see that module's docstring for why the split is a real boundary
rather than tidiness.
"""

import torch
import torch.nn as nn

from grace_adapter.models.adapter import GatedResidualAdapter
from grace_adapter.models.discrepancy import DiscrepancyHead
from grace_adapter.models.severity import SeverityHead
from eval.splits.base import FeatureSpec


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


def build_adapter(
    spec: FeatureSpec,
    cfg,
    tap_spec: FeatureSpec | None = None,
    tap_names=(),
) -> GatedResidualAdapter:
    """The plain adapter, or a `LadderAdapter` when a tap spec is given.

    Dispatch is on `tap_spec is not None` rather than on `cfg.taps`, because the
    tap spec comes from the split and the config only expresses an intent. The
    two are reconciled once, in `grace_adapter.models.ladder.tap_spec_for`;
    everywhere else the presence of a spec *is* the decision.
    """
    if tap_spec is not None:
        from grace_adapter.models.ladder import build_ladder

        return build_ladder(spec, tap_spec, cfg, tap_names)
    return GatedResidualAdapter(
        dim=spec.dim,
        gate_shape=gate_shape_for(spec, cfg.per_channel_gate),
        bottleneck=cfg.bottleneck,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        severity_film=cfg.severity_film,
        gate_init=cfg.gate_init,
    )


def build_severity_head(spec: FeatureSpec, hidden: int = 256) -> SeverityHead:
    return SeverityHead(dim=spec.dim, hidden=hidden)


def build_discrepancy_head(spec: FeatureSpec, cfg, n_taps: int = 0) -> DiscrepancyHead:
    """`n_taps` comes from the stage-1 adapter, not from `cfg`.

    `cfg.use_taps` is an intent; how many taps there are is a fact about the
    frozen checkpoint stage 2 was pointed at. Same rule as `build_adapter`'s
    dispatch on `tap_spec is not None`.
    """
    return DiscrepancyHead(
        spec=spec, hidden=cfg.hidden, proj=cfg.proj,
        use_severity=cfg.use_severity, n_taps=n_taps,
    )


def save_adapter(path, adapter: nn.Module, spec: FeatureSpec, cfg, extra: dict | None = None):
    """Weights *and* the config that shaped them.

    An adapter checkpoint must be loadable by the eval harness with no reference
    to the training run that produced it -- that is what lets
    `eval/configs/detectors/*+grace.yaml` name a checkpoint and nothing else.
    """
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "feature_spec": spec.to_dict(),
            "adapter_cfg": vars(cfg),
            # Read off the adapter, not off the config: `cfg.taps` is an intent
            # and the built module is the fact. A checkpoint has to rebuild into
            # the same class it was saved from with no reference to the run.
            **_tap_payload(adapter),
            **(extra or {}),
        },
        path,
    )


def _tap_payload(adapter: nn.Module) -> dict:
    if not getattr(adapter, "reads_taps", False):
        return {}
    return {
        "tap_spec": FeatureSpec(
            layout="layers", shape=(adapter.n_taps, adapter.tap_in)
        ).to_dict(),
        "tap_names": list(adapter.tap_names),
    }


def load_adapter(
    checkpoint: str,
    spec: FeatureSpec | None = None,
    tap_spec: FeatureSpec | None = None,
) -> GatedResidualAdapter:
    """Rebuild from the checkpoint's stored config and load its weights.

    If `spec` is given it is checked against the stored one: loading an adapter
    trained on a different feature layout would otherwise fail deep inside a
    matmul rather than here. `tap_spec` is checked the same way -- a ladder
    scored against a split tapping a different number of blocks is the tap-set
    equivalent of the wrong feature layout, and the mismatch that would
    otherwise produce plausible numbers for a model reading the wrong part of
    the trunk.
    """
    from train.config import AdapterConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = FeatureSpec.from_dict(payload["feature_spec"])
    if spec is not None and (spec.layout, spec.shape) != (stored.layout, stored.shape):
        raise ValueError(
            f"{checkpoint} was trained on {stored.layout}{stored.shape} but this "
            f"detector emits {spec.layout}{spec.shape}"
        )

    stored_taps = payload.get("tap_spec")
    stored_taps = FeatureSpec.from_dict(stored_taps) if stored_taps else None
    if tap_spec is not None and stored_taps is None:
        raise ValueError(
            f"{checkpoint} is a plain adapter but this split emits taps "
            f"{tap_spec.shape}. Scoring it would silently ignore the ladder the "
            f"split was configured for -- name a ladder checkpoint, or drop "
            f"`tap_blocks` from the split args."
        )
    if stored_taps is not None and tap_spec is not None and tap_spec.shape != stored_taps.shape:
        raise ValueError(
            f"{checkpoint} is a ladder over {stored_taps.shape[0]} tap(s) "
            f"{payload.get('tap_names')} but this split emits {tap_spec.shape[0]}. "
            f"The tap sets must match exactly."
        )

    adapter = build_adapter(
        stored,
        AdapterConfig(**payload["adapter_cfg"]),
        tap_spec=stored_taps,
        tap_names=tuple(payload.get("tap_names", ())),
    )
    adapter.load_state_dict(payload["state_dict"])
    return adapter.eval()
