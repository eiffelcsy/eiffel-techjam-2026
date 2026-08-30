"""DINOv3 split -- the proof-of-concept seam, and the only one that runs today.

The only one at all, now that the zoo splits are gone. They each reconstructed
a seam inside a vendored repo that was never in this tree, and RINE's head
composition carried a standing warning that it must be checked against its
clone. That is the right design for adapting somebody else's published
detector, and it was a bad way to find out whether GRACE works at all: a
retention number from a wrongly composed head is a comparison against a model
that was never benchmarked, and there is no way to tell that from the curve.

`eval.detectors.dinov3.DINOv3MLPDetector` is built with the seam already in
it, so this class delegates rather than reconstructs:

    trunk -> detector.trunk    frozen DINOv3 ViT-S/16, pooled -> (B, D)
    head  -> detector.head     LayerNorm -> MLP -> one logit

`head(trunk(x)) == detector(x)` is then true by construction -- `forward` on the
detector is literally `self.head(self.trunk(x))`. `verify_split` still runs,
because "true by construction" is a claim about code that someone will edit.

Layout is `vector`, deliberately
--------------------------------
The pooled descriptor is one embedding per image, so the adapter's gate is a
single `(D,)` vector and there is no per-block damage profile to plot. That
forfeits the per-layer interpretability figure a `layers` seam would give, and
it is the right trade for a PoC: it makes the cache 768 bytes per image per view instead of 48 KB (a factor
of 64), which is what lets the whole pipeline -- render, stage 1, stage 2, eval --
run end to end in minutes on a laptop with no GPU.

The `layers` variant is a small change when it is wanted: emit the per-block CLS
tokens from `output_hidden_states=True` as `(B, 12, 384)` and give the head a
RINE-style importance weighting over them. Everything downstream of the split
already handles that layout -- `grace.models.factory` picks the `(L, D)` gate off
`FeatureSpec.layout` alone.

Taps, for the ladder
--------------------
That `layers` variant would change *the detector*: a head reading all twelve
blocks is a different model with a different baseline. Taps are the other thing,
and they change nothing. The seam stays exactly where it was, the head still
reads one pooled vector, and the intermediate blocks are handed to the adapter as
**side information about the damage** -- see `grace.models.ladder`.

`trunk` throws these away today: the ViT computes all thirteen hidden states and
the pooled output keeps one. `trunk_with_taps` keeps a few more, for free.
"""

import torch

from grace.splits.base import FeatureSpec, SplitDetector
from grace.splits.verify import verify_split, verify_taps

DEFAULT_TAP_BLOCKS = (0, 2, 4, 6, 9)
"""Which hidden states to tap, indexed as `output_hidden_states` numbers them:
**0 is the patch embedding**, 1..12 are the twelve block outputs.

Chosen by measurement, not by spacing. The question a ladder has to answer is
"which stage did the damage enter at", so the tap set was scored on exactly
that: identify which of nine L1 transforms hit an image, from the per-block drift
profile alone (60 NTIRE-val images, crop preprocessing, 5-fold CV, chance
0.111).

    what the adapter can read              shape only   with magnitude
    seam only (the plain adapter)               0.137            0.376
    CLS, all 13 hidden states                   0.748            0.874
    patchmean, all 13 hidden states             0.748            0.861
    cls+patchmean, blocks (0, 2, 4, 6, 9)       0.822            0.896

Three things that table settles, all of them against the obvious guess:

  * **Not every block.** Five taps match or beat all thirteen at 5/13 the
    storage. The profiles separate in the first third -- brightness peaks at the
    embedding, blur at block 2, resize and noise at blocks 3-4, while JPEG rises
    monotonically to the end -- and past block 6 the curves run near-parallel.
  * **Not CLS alone.** CLS at every block loses to cls+patchmean at five. The
    pooling here mirrors the seam's (`detector.pool_tokens`), which is both the
    forensically right axis -- see POOLS in `eval.detectors.dinov3`, the
    traces are local -- and what makes a tap at the last block reduce to the seam
    exactly.
  * **The seam is nearly blind to this.** 0.376 against 0.896 is the ladder's
    entire justification: the correction the plain adapter proposes has to infer
    the damage from the output, and it mostly cannot.

Block 12 is deliberately absent: pooled, it *is* the seam feature, so tapping it
would spend a fifth of the cache restating an input the adapter already has.
`tests/test_dinov3_taps.py` taps it on purpose to check that identity holds.
"""


class DINOv3Split(SplitDetector):
    """Cut the DINOv3 probe detector at its pooled feature vector.

    Parameters
    ----------
    tap_blocks : hidden-state indices for the ladder, or None/() for no taps.
                 See DEFAULT_TAP_BLOCKS. Passing this changes what the split
                 *emits*, never what the detector computes: `trunk`, `head` and
                 every number they produce are untouched.
    """

    def __init__(self, detector, verify: bool = True, tap_blocks=None):
        super().__init__(detector)
        for attr in ("trunk", "head", "feature_dim"):
            if not hasattr(detector, attr):
                raise TypeError(
                    f"DINOv3Split expects a DINOv3MLPDetector (it needs .{attr}); "
                    f"got {type(detector).__name__}. This split delegates to a "
                    f"detector that already exposes its seam -- it does not "
                    f"reconstruct one."
                )
        # Not an error: rendering a cache needs only the trunk, and the trunk is
        # frozen pretrained weights whatever the head holds. Scoring with an
        # untrained head, on the other hand, produces a ~0.5 AUC that looks like
        # a failed adapter rather than a missing stage 0.
        if getattr(detector, "head_untrained", False):
            import warnings

            warnings.warn(
                f"{detector.name} has a randomly initialized head "
                "(head_checkpoint: null). Trunk features are still valid, so a "
                "cache rendered from this is fine -- but any logit, AUC or "
                "head-Jacobian computed from it is noise. Run "
                "scripts/train_probe.py first.",
                stacklevel=2,
            )
        # Read off the model, never hardcoded: hidden_states is one longer than
        # the block count because index 0 is the patch embedding.
        self._n_states = detector.backbone.config.num_hidden_layers + 1
        self.tap_blocks = tuple(int(b) for b in (tap_blocks or ()))
        bad = [b for b in self.tap_blocks if not 0 <= b < self._n_states]
        if bad:
            raise ValueError(
                f"tap_blocks {bad} out of range: this backbone has "
                f"{self._n_states} hidden states (0 = patch embedding, "
                f"1..{self._n_states - 1} = block outputs)."
            )
        if len(set(self.tap_blocks)) != len(self.tap_blocks):
            raise ValueError(f"tap_blocks has duplicates: {self.tap_blocks}")

        if verify:
            verify_split(self)
            verify_taps(self)

    @property
    def feature_spec(self) -> FeatureSpec:
        # Read off the detector, never hardcoded: `pool` changes the width by 2x
        # and a ViT-B mirror changes it again.
        return FeatureSpec(layout="vector", shape=(self.detector.feature_dim,))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.detector.trunk(x)

    def head(self, f: torch.Tensor) -> torch.Tensor:
        return self.detector.head(f)

    def taps(self) -> tuple[str, ...]:
        """`("block00", "block02", ...)` -- the hidden-state index, zero-padded.

        Named by what was read rather than by position, so `spec.json` still
        answers "which block is tap 3" after someone edits `tap_blocks`.
        """
        return tuple(f"block{b:02d}" for b in self.tap_blocks)

    def tap_spec(self) -> FeatureSpec | None:
        """`(K, feature_dim)` -- taps are pooled exactly as the seam is."""
        if not self.tap_blocks:
            return None
        return FeatureSpec(
            layout="layers", shape=(len(self.tap_blocks), self.detector.feature_dim)
        )

    def trunk_with_taps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One forward pass; keep the hidden states `trunk` discards.

        **Every tap goes through the backbone's final norm**, and the reason is
        arithmetic rather than taste. In this model
        `last_hidden_state == backbone.norm(hidden_states[-1])` exactly, so
        applying the same norm to every hidden state is what makes a tap at the
        last block equal the seam -- the `verify_taps` identity. It is also what
        makes the taps *comparable to each other*: raw activations grow from
        ~5.8 to ~1691 max-abs across depth on this backbone, a 290x spread that
        the shared tap projection in `grace.models.ladder` would otherwise have
        to absorb, and that float16 storage has no reason to spend range on.
        Normalized, every tap sits in ~9-25.
        """
        backbone = self.detector.backbone
        need_taps = bool(self.tap_blocks)
        out = backbone(pixel_values=x, output_hidden_states=need_taps)
        f = self.detector.pool_tokens(out.last_hidden_state)
        if not need_taps:
            return f, None
        taps = torch.stack(
            [
                self.detector.pool_tokens(backbone.norm(out.hidden_states[b]))
                for b in self.tap_blocks
            ],
            dim=1,
        )
        return f, taps

    def head_modules(self) -> dict:
        """For `verify_split`'s failure message. One module, by construction."""
        return {"head_module": self.detector.head_module}
