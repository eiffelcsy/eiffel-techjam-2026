"""The frequency-enriched detector -- how the DCT branch re-enters the harness.

    GRACE        logit = head(adapter(trunk(x)))
    GRACE-freq   logit = head(enricher(adapter(trunk(x)), dct(x)))

A `FrozenDetector` like `AdaptedDetector`, named in a run config by dotted import
path, scored by the Day-1 harness with the same conditions, threshold rule,
retention denominator and JSON schema. It differs from every other detector in
this tree in exactly one respect: it reads the image twice.

WHY IT HAS TO. The trunk is fed a 224px normalized tensor. Generation traces are
a local high-frequency phenomenon at native pixel scale, so whatever the resize
destroyed is not in that tensor and no module downstream of it can recover the
information -- not at any capacity, not in principle. The DCT branch therefore
reads the PIL image, in the worker, beside the preprocessing. `aux_fn()` is what
carries it there; `preprocessing.dataset.Inputs` is what carries it back.

THE TWO BRANCHES MUST SEE ONE WINDOW. This is the invariant the whole method
rests on, and it is the one that is easy to break silently, because the two
branches choose their window in different places:

    training   the dataset draws a 128-S_max window (`multiscale_crop`) and both
               branches are handed the result. One variable, nothing to align.
    eval       the window is chosen INSIDE preprocessing (`_CropResizePreprocess`,
               `_ResamplePreprocess`), from which no tensor can be read back to a
               window. So the arm's geometry is restated on the frequency side,
               derived from the same `dinov3.VIEWS` table -- not from a config
               key that could disagree with it.

`tests/test_fused_split.py` pins the two to the same pixels, and
`_freq_geometry` below is the only place the restatement happens.

THREE CONFIGURATIONS, all this one class:

    enricher: null                      the NULL enricher -- freshly initialized,
                                        so identity by construction. It must
                                        reproduce the `+grace` arm to the last
                                        decimal (E10), exactly as
                                        `dinov3+identity.yaml` does for stage 1.
    enricher: .../enricher.pt           GRACE-freq
    + a checkpoint carrying a head      E14's fine-tuned-head arm; the head
                                        travels inside the enricher checkpoint
                                        because the two are only meaningful
                                        together.
"""

import torch
from PIL import Image

from train.config import EnricherConfig, FreqConfig
from grace_adapter.models.factory import load_adapter
from freq_branch.models.factory import build_enricher, load_enricher
from grace_adapter.models.severity import SeverityHead
from eval.splits import build_split
from eval.config import load_detector_config
from eval.detectors import build_detector
from eval.detectors.base import FrozenDetector
from eval.detectors.dinov3 import VIEWS
from freq_branch.view import FreqExtract


def _freq_geometry(base_detector) -> tuple[str, int]:
    """The window the frequency branch must take, from the arm's `input_mode`.

    Read off `VIEWS`, the same table `_build_preprocess` reads, so the spatial
    and frequency branches cannot drift onto different windows through two
    config keys that were meant to agree. A mode not in `VIEWS` -- `multiscale`,
    `resize`, `crop` -- means the window was already chosen upstream and this
    reads what it is handed.
    """
    mode = getattr(base_detector, "input_mode", "resize")
    if mode not in VIEWS:
        return "", 0
    geometry, size = VIEWS[mode]
    return geometry, size


class FusedDetector(FrozenDetector):
    """A frozen detector, a GRACE adapter, and a frequency enricher.

    Parameters
    ----------
    base        : path to the base detector's config (or the same mapping inline)
    split       : dotted path to the SplitDetector for that base
    checkpoint  : stage-1 adapter weights. None = identity adapter, which makes
                  this arm "the enricher alone, on uncorrected features" -- a
                  control, not the headline.
    enricher    : stage-2 enricher weights. None builds a FRESH enricher, which
                  is exactly the identity and is E10's null arm.
    freq        : the DCT protocol, as a `FreqConfig`-shaped mapping. Ignored
                  when `enricher` names a checkpoint, which carries its own --
                  the coefficient axis has to mean what the band masks were
                  trained against, and the checkpoint is the authority on that.
    """

    def __init__(
        self,
        base,
        split: str,
        checkpoint: str | None = None,
        enricher: str | None = None,
        freq: dict | None = None,
        enricher_args: dict | None = None,
        name: str = "grace-freq",
        split_args: dict | None = None,
    ):
        super().__init__()
        detector = build_detector(load_detector_config(base))
        self.split = build_split(detector, split, **(split_args or {}))
        self.name = name

        spec = self.split.feature_spec
        tap_spec = self.split.tap_spec()
        self.adapter = load_adapter(checkpoint, spec, tap_spec) if checkpoint else None
        self.severity_head = None
        if self.adapter is not None and self.adapter.film is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "severity_state_dict" in payload:
                self.severity_head = SeverityHead(spec.dim)
                self.severity_head.load_state_dict(payload["severity_state_dict"])

        if enricher is not None:
            payload = torch.load(enricher, map_location="cpu", weights_only=False)
            self.freq_cfg = FreqConfig(**payload["freq_cfg"])
            self.enricher = load_enricher(enricher, spec, self.freq_cfg.feature())
            self._load_finetuned_head(payload, enricher)
        else:
            # The null arm. A FRESH enricher, not a missing one: skipping the
            # module entirely would test that `FusedDetector` can be bypassed,
            # where E10 asks whether the module -- built, wired, forward-passed --
            # is the identity at initialization. Those are different claims and
            # only the second one is worth a hard stop in `run_all.sh`.
            self.freq_cfg = FreqConfig(**{**{"enabled": True}, **(freq or {})})
            self.enricher = build_enricher(
                spec, self.freq_cfg.feature(),
                EnricherConfig(**(enricher_args or {})),
                patch=self.freq_cfg.patch, channels=self.freq_cfg.channels,
            )

        geometry, size = _freq_geometry(detector)
        self._aux = FreqExtract(
            patch=self.freq_cfg.patch, grid=self.freq_cfg.grid,
            radial=self.freq_cfg.radial, geometry=geometry, size=size,
        )
        self.freeze()

    def _load_finetuned_head(self, payload, path: str) -> None:
        """E14's head, when the checkpoint carries one.

        Loaded into the split's own head module rather than kept alongside it:
        the fine-tuned head IS this detector's head, and a second code path for
        "which head do I score with" is one more place for the frozen arm and the
        fine-tuned arm to differ by something other than the swept key.

        A checkpoint that claims a fine-tuned head and carries no weights is
        refused rather than silently scored with the frozen one, because the
        difference is the entire experiment.
        """
        state = payload.get("head_state_dict")
        if not payload.get("finetune_head"):
            if state is not None:
                raise ValueError(
                    f"{path} carries head weights but finetune_head is false. "
                    f"Which head this detector scores with would be ambiguous."
                )
            return
        if state is None:
            raise ValueError(
                f"{path} says finetune_head but carries no head weights -- it "
                f"would score with the frozen head under the fine-tuned arm's "
                f"name, which is the one comparison E14 exists to make."
            )
        self.split.head_module().load_state_dict(state)

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.split.detector.preprocess(img)

    def preprocess_fn(self):
        return self.split.preprocess_fn()

    def aux_fn(self):
        """The DCT read, run in the worker beside the preprocessing.

        Picklable and model-free by construction -- five ints and two strings --
        which is the same contract `preprocess_fn` is held to and for the same
        reason: the dataset is forked into DataLoader workers.
        """
        return self._aux

    def forward(self, x) -> torch.Tensor:
        """`x` is an `Inputs` -- the image tensor and its frequency tokens.

        A bare tensor is refused rather than defaulted, because the only way it
        arrives here is a caller that bypassed `aux_fn`, and the default would be
        to score the enricher over zeros and report it as GRACE-freq.
        """
        if not hasattr(x, "aux"):
            raise TypeError(
                f"{type(self).__name__} needs (image, frequency) inputs but got a "
                f"bare {type(x).__name__}. The dataset builds them from "
                f"`detector.aux_fn()` -- see preprocessing.dataset.Inputs."
            )
        f, taps = self.split.trunk_with_taps(x.x)
        f = f.float()
        if self.adapter is None:
            corrected, severity = f, None
        else:
            taps = taps.float() if taps is not None and self.adapter.reads_taps else None
            severity = self.severity_head(f) if self.severity_head is not None else None
            corrected = self.adapter(f, severity=severity, taps=taps)
        return self.split.head(self.enricher(corrected, x.aux.float(), severity))
