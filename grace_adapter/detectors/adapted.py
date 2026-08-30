"""The adapted detector -- how GRACE re-enters the evaluation harness.

    GRACE     logit = head(adapter(trunk(x)))
    GRACE-D   logit = head(adapter(trunk(x))) + β · aux(Δ, severity)

`AdaptedDetector` is a `FrozenDetector`, so it is named in a run config by dotted
import path like any other model and the Day-1 harness scores it with no change
whatsoever: same conditions, same threshold rule, same retention denominator,
same JSON schema, same report table. The baseline and the adapted model differ by
one config file, which is the plug-and-play claim in concrete form.

`split_args` reaches the split, and is how a ladder checkpoint is told which
blocks to tap at inference: `{tap_blocks: [0, 2, 4, 6, 9]}`, matching whatever
the cache it was trained on was rendered with. Mismatches are refused by
`load_adapter` rather than silently scored.

Three configurations, all the same class:

    checkpoint: null                    the null adapter -- must reproduce the
                                        baseline to the last decimal (E1)
    checkpoint: .../ema.pt              GRACE, label-free
    + discrepancy: .../discrepancy.pt   GRACE-D, the supervised variant

Preprocessing is delegated to the base detector unchanged, `preprocess_fn`
included: the dataset is forked into DataLoader workers and must not carry the
model, and standardising preprocessing across the zoo would break the baselines
being compared against.
"""

import torch
from PIL import Image

from grace_adapter.models.discrepancy import FusedHead
from grace_adapter.models.factory import build_discrepancy_head, load_adapter
from grace_adapter.models.severity import SeverityHead
from eval.splits import build_split
from eval.config import load_detector_config
from eval.detectors import build_detector
from eval.detectors.base import FrozenDetector


class AdaptedDetector(FrozenDetector):
    """A frozen detector with a trained GRACE adapter spliced at its seam.

    Parameters
    ----------
    base        : path to the base detector's config (or the same mapping inline)
    split       : dotted path to the SplitDetector for that base
    checkpoint  : trained adapter weights; None = identity, i.e. exactly the base
                  detector. Run that first -- it should reproduce the Day-1
                  numbers exactly, and if it does not, the split is wrong and
                  every comparison downstream is against a model that was never
                  benchmarked.
    discrepancy : stage-2 checkpoint. Present = GRACE-D.
    """

    def __init__(
        self,
        base,
        split: str,
        checkpoint: str | None = None,
        discrepancy: str | None = None,
        name: str = "grace",
        split_args: dict | None = None,
    ):
        super().__init__()
        detector = build_detector(load_detector_config(base))
        self.split = build_split(detector, split, **(split_args or {}))
        self.name = name

        spec = self.split.feature_spec
        # Passed to `load_adapter` so a ladder checkpoint scored against a split
        # tapping other blocks -- or a plain checkpoint against a tapping split
        # -- is refused here rather than producing plausible numbers for a model
        # reading the wrong part of the trunk.
        tap_spec = self.split.tap_spec()
        self.adapter = load_adapter(checkpoint, spec, tap_spec) if checkpoint else None
        self.severity_head = None
        self.fused = None

        if self.adapter is not None and self.adapter.film is not None:
            # The severity head ships inside the stage-1 checkpoint; without it
            # the FiLM path has no input and conditioning silently reverts to the
            # unconditioned gate.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "severity_state_dict" in payload:
                self.severity_head = SeverityHead(spec.dim)
                self.severity_head.load_state_dict(payload["severity_state_dict"])

        if discrepancy is not None:
            if self.adapter is None:
                raise ValueError("a discrepancy head needs an adapter to produce Δ")
            payload = torch.load(discrepancy, map_location="cpu", weights_only=False)
            cfg = type("_C", (), payload["discrepancy_cfg"])()
            self.fused = FusedHead(build_discrepancy_head(spec, cfg))
            self.fused.load_state_dict(payload["state_dict"])
            if self.fused.aux.use_severity and self.severity_head is None:
                raise ValueError(
                    f"{discrepancy} was trained with use_severity=True but "
                    f"{checkpoint} carries no severity head. Retrain stage 1 with "
                    f"lam_sev > 0, or stage 2 with use_severity: false."
                )

        self.freeze()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.split.detector.preprocess(img)

    def preprocess_fn(self):
        return self.split.preprocess_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One trunk pass either way: `trunk_with_taps` returns `(f, None)` on a
        # split with no taps, so the ladder costs nothing at inference beyond
        # the tap projections themselves. This is the property that makes the
        # ladder deployable at all -- the taps are activations the forward pass
        # already produced.
        f, taps = self.split.trunk_with_taps(x)
        if self.adapter is None:
            return self.split.head(f)

        f = f.float()
        taps = taps.float() if taps is not None and self.adapter.reads_taps else None
        severity = self.severity_head(f) if self.severity_head is not None else None

        f_adapted = self.adapter(f, severity=severity, taps=taps)
        logit = self.split.head(f_adapted)

        if self.fused is not None:
            logit = self.fused(logit, f_adapted - f, severity)
        return logit
