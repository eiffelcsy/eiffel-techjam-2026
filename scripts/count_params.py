import json
import os

import torch


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def report(name, m):
    total = n_params(m)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:40s} params={total:>12,d}  trainable={trainable:>12,d}")


def load_dict(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    root = os.getcwd()

    backbone_ckpt = os.path.join(
        root, "checkpoints", "probe", "dinov3_wildfake_multiscale", "head.pt"
    )
    adapter_ckpt = os.path.join(root, "checkpoints", "grace", "dinov3_multiscale", "ema.pt")
    enricher_ckpt = os.path.join(root, "checkpoints", "grace", "dinov3_enrich", "enricher.pt")

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        "facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    report("DINOv3 ViT-S/16 trunk", backbone)

    from eval.splits.base import FeatureSpec

    head_payload = load_dict(backbone_ckpt)
    print("head payload keys:", sorted(head_payload.keys()))
    print("  head cfg:", {k: head_payload.get(k) for k in ("hidden", "n_layers", "dropout", "pool", "feature_dim")})

    from eval.detectors.dinov3 import ProbeHead

    head = ProbeHead(
        head_payload["feature_dim"],
        hidden=head_payload["hidden"],
        n_layers=head_payload["n_layers"],
        dropout=head_payload.get("dropout", 0.0),
    )
    head.load_state_dict(head_payload["state_dict"])
    report("ProbeHead (MLP classifier head)", head)

    from train.config import AdapterConfig, EnricherConfig, FreqConfig
    from grace_adapter.models.factory import load_adapter
    from freq_branch.models.factory import load_enricher

    spec = FeatureSpec(layout="vector", shape=(head_payload["feature_dim"],), dtype="float16")
    print("feature_spec:", spec.to_dict())

    adapter_payload = load_dict(adapter_ckpt)
    print("adapter payload keys:", sorted(adapter_payload.keys()))
    print("  adapter_cfg:", adapter_payload.get("adapter_cfg"))
    adapter = load_adapter(adapter_ckpt, spec)
    report("GatedResidualAdapter", adapter)

    severity = None
    if adapter.film is not None and "severity_state_dict" in adapter_payload:
        from grace_adapter.models.severity import SeverityHead

        severity = SeverityHead(spec.dim)
        severity.load_state_dict(adapter_payload["severity_state_dict"])
        report("SeverityHead", severity)

    enricher_payload = load_dict(enricher_ckpt)
    print("enricher payload keys:", sorted(enricher_payload.keys()))
    print("  enricher_cfg:", enricher_payload.get("enricher_cfg"))
    print("  aux_cfg:", enricher_payload.get("aux_cfg"))
    print("  freq_cfg:", enricher_payload.get("freq_cfg"))
    freq_cfg = FreqConfig(**enricher_payload["freq_cfg"])
    enricher = load_enricher(enricher_ckpt, spec, freq_cfg.feature())
    report("FrequencyEnricher", enricher)

    from freq_branch.models.frequency import EnricherFusedLogit
    fused = None
    if enricher_payload.get("fused_logit_state_dict"):
        aux_cfg = enricher_payload.get("aux_cfg") or {}
        fused = EnricherFusedLogit(spec.dim, hidden=aux_cfg.get("hidden", 128))
        fused.load_state_dict(enricher_payload["fused_logit_state_dict"])
        report("EnricherFusedLogit (aux head + beta)", fused)

    total = n_params(backbone) + n_params(head)
    components = {"DINOv3 trunk": backbone, "ProbeHead": head}
    if severity is not None:
        total += n_params(severity)
        components["SeverityHead"] = severity
    total += n_params(adapter)
    components["Adapter"] = adapter
    total += n_params(enricher)
    components["Enricher"] = enricher
    if fused is not None:
        total += n_params(fused)
        components["FusedLogit"] = fused

    print("=" * 80)
    print(f"{'TOTAL grace-freq pipeline':40s} params={total:>12,d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
