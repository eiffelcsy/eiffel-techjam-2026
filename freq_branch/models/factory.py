"""FeatureSpec -> the frequency enricher. The sibling of `grace_adapter.models.factory`.

Its own build/save/load rather than fields on the adapter's, and that is a
deliberate boundary rather than tidiness. `grace_adapter.models.factory.load_adapter`
rebuilds a stage-1 module with `AdapterConfig(**payload["adapter_cfg"])`, so
every field on that dataclass is a checkpoint-compatibility surface: a new one
there breaks every adapter checkpoint already on disk. The enricher is a
separate module trained in a separate stage against a separate objective, so it
gets a separate checkpoint -- following `discrepancy.pt`, for the same reason.
"""

import torch
import torch.nn as nn

from eval.splits.base import FeatureSpec
from freq_branch.models.frequency import FrequencyEnricher


def build_enricher(spec: FeatureSpec, freq: FeatureSpec, cfg, patch: int = 8,
                   channels: int = 3) -> FrequencyEnricher:
    """`patch`/`channels` come from the FreqConfig the cache was rendered under.

    They are not derivable from `freq.shape` alone -- `(196, 192)` is 3 channels
    of 8x8 and also 1 channel of 192 coefficients, and the band masks and the
    top-k selection index the axis differently in each case. So they travel with
    the render protocol rather than being inferred from a shape.
    """
    n_cells, n_coeffs = freq.shape
    return FrequencyEnricher(
        dim=spec.dim, n_cells=n_cells, n_coeffs=n_coeffs,
        patch=patch, channels=channels,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_bands=cfg.n_bands,
        dropout=cfg.dropout, gate_init=cfg.gate_init,
        severity_film=cfg.severity_film, pos_emb=cfg.pos_emb,
        top_k=cfg.top_k, learn_masks=cfg.learn_masks,
    )


def save_enricher(path, enricher: nn.Module, spec: FeatureSpec, freq: FeatureSpec,
                  cfg, freq_cfg, extra: dict | None = None) -> None:
    """Weights, both specs, both configs, and whatever the run wants beside them.

    Both configs, because rebuilding needs both: `enricher_cfg` shapes the
    module and `freq_cfg` says what its coefficient axis MEANS. A checkpoint
    that carried only the first would load cleanly against a cache rendered at a
    different patch size and attend over the wrong frequencies in silence.
    """
    torch.save(
        {
            "state_dict": enricher.state_dict(),
            "feature_spec": spec.to_dict(),
            "freq_spec": freq.to_dict(),
            "enricher_cfg": vars(cfg),
            "freq_cfg": vars(freq_cfg),
            **(extra or {}),
        },
        path,
    )


def load_enricher(
    checkpoint: str,
    spec: FeatureSpec | None = None,
    freq: FeatureSpec | None = None,
) -> FrequencyEnricher:
    """Rebuild from the checkpoint's own configs and load its weights.

    Both specs are checked when given, for the reason
    `grace_adapter.models.factory.load_adapter` checks `feature_spec`: a
    mismatch would otherwise fail deep inside a matmul, or -- worse, and this is
    the frequency-specific case -- not fail at all, because `(196, 192)` from an
    8x8x3 render and `(196, 192)` from some other geometry are the same shape
    and different features.
    """
    from train.config import EnricherConfig, FreqConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = FeatureSpec.from_dict(payload["feature_spec"])
    if spec is not None and (spec.layout, spec.shape) != (stored.layout, stored.shape):
        raise ValueError(
            f"{checkpoint} was trained on {stored.layout}{stored.shape} but this "
            f"detector emits {spec.layout}{spec.shape}"
        )
    stored_freq = FeatureSpec.from_dict(payload["freq_spec"])
    if freq is not None and stored_freq.shape != freq.shape:
        raise ValueError(
            f"{checkpoint} was trained on frequency tokens {stored_freq.shape} but "
            f"this run supplies {freq.shape}. Same DCT protocol or re-render."
        )

    freq_cfg = FreqConfig(**payload["freq_cfg"])
    enricher = build_enricher(
        stored, stored_freq, EnricherConfig(**payload["enricher_cfg"]),
        patch=freq_cfg.patch, channels=freq_cfg.channels,
    )
    enricher.load_state_dict(payload["state_dict"])
    return enricher.eval()
