"""DINOv3 ViT-S/16 + a linear-probe MLP head -- the detector GRACE's PoC adapts.

The ONLY detector in this package now, and it is built here rather than
downloaded, for one reason: **GRACE needs a detector whose trunk/head seam is
not in dispute.**

The published zoo (B-Free, GAPL, RINE) each reconstructed a seam inside a repo
cloned by hand under `third_party/`, and RINE's head composition was never
verified against its clone. All of it has been removed. What that costs is the
cross-detector evidence: GRACE can now only be demonstrated on this one seam,
so "the method generalizes across detectors" is no longer a claim this tree can
support. What it buys is that everything here runs end to end with no clone.

Here the seam is a construction rather than a reconstruction:

    trunk = frozen DINOv3 ViT-S/16, pooled     -> (B, D)   layout "vector"
    head  = LayerNorm -> MLP -> one logit      -> (B,)

so `head(trunk(x)) == detector(x)` holds by definition, `DINOv3Split` has nothing
to guess, and E1 (the identity adapter reproducing the baseline exactly) is a
tautology instead of a nail-biter. It is also small: 21M frozen parameters and a
768-dim feature, so the whole cache for a 200-image PoC is under 3 MB and stage 1
trains on a laptop in seconds.

**The backbone is the distilled ViT-S/16.** `facebook/dinov3-vits16-pretrain-
lvd1689m` is the model distilled from the ViT-7B teacher on LVD-1689M, which is
why a 21M-parameter trunk carries features worth correcting in the first place.

**The head is trained on CLEAN images only**, by `grace_adapter/scripts/train_probe.py`.
That is not a shortcut, it is the premise: GRACE's whole claim is about a
detector that was fit on clean data and whose accuracy collapses under
degradation. A head trained with degradation augmentation would have already
solved the problem GRACE is trying to solve, and its retention curve would say
nothing.

The trunk never sees the head, so a cache rendered from this detector stays valid
across probe retrainings -- `train.cache.spec` hashes the detector's *config*,
and the head checkpoint path in it names weights the cached features do not
depend on.

Gated weights
-------------
`facebook/dinov3-*` is a gated Hub repo. Accept the licence on the model page and
`hf auth login` once; or point `backbone_id` at any mirror you already have.
Nothing else in this file assumes the official id.
"""

import torch
import torch.nn as nn
from PIL import Image

from eval.detectors.base import FrozenDetector
from eval.detectors.hf import (
    _CropPreprocess, _CropResizePreprocess, _ProcessorPreprocess, _ResamplePreprocess,
)

DEFAULT_BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"

INPUT_MODES = ("resize", "crop", "multiscale", "crop200", "resample512")
"""How a source image becomes the model's 224x224 input.

    resize       the processor's own transform: squash the whole image to 224
    crop         center 224x224 window at the source's native resolution
    multiscale   the TRAINING protocol: the dataset already handed us a random
                 128-512px window, so this squashes that window to 224
    crop200      EVAL ARM (a): centre 200x200 window at native scale, then to 224
    resample512  EVAL ARM (b): whole image to 512x512, then to 224

The last three exist because WildFake ships its COCO reals pre-resized to a
uniform 200x200 while its DALL-E 3 fakes are native 1024px output, so on the
reported benchmark `max(w, h)` separates the classes at AUC 1.0000 -- measured
over all 13,841 rows, not estimated. A model shown whole images reads that
instead of the content, and a frequency branch reads it hardest.

`multiscale` is where the fix lives. The crop is drawn in the dataset rather
than here, seeded on `(index, epoch)` by `preprocessing.degrade.crop.multiscale_crop`
so the render stays reproducible and `sha_preprocess` still sees a deterministic
transform; by the time preprocessing runs, the window is already chosen and all
that is left is the squash to 224. The mode name is therefore not describing a
different transform from `resize` -- it is recording which protocol the head was
fit under, which is exactly what `_assert_head_matches` needs to tell "resize of
a whole image" apart from "resize of a native-scale crop".

The two eval arms give every image identical dimensions within an arm, so the
dimension shortcut is 0.5 by construction rather than by normalisation. Arm (a)
is in distribution for a multi-scale-trained model (a fixed 200px window is a
special case of the training draw) and is the informative one. Arm (b) is out of
distribution -- training never squashes a whole image -- so read it as a
robustness check rather than as a like-for-like comparison.

`resize` is what the processor config asks for (`size: {224, 224}`,
`do_center_crop: null`, `default_to_square: true`) and it is the right default
for DINOv3's *intended* use, where the task is semantic and seeing the whole
frame matters.

It is the wrong default here, and measurably so. The finding that produced this
flag was made on SID_Set, the dataset this project used before NTIRE, and it is
recorded because the mechanism is a property of the preprocessing rather than of
that dataset: under `resize`, a probe reached 0.9999 val AUC and then held 0.985
AUC after the image was downscaled to 32x32 and back -- a resolution at which no
generation trace of any kind survives. That is not a forensic detector; it is a
content classifier, and it cannot show the degradation collapse GRACE exists to
repair (retention stayed at 100% through JPEG-30, blur sigma=2.0 and 4x
downscale).

The 32x32 round trip is worth keeping as a standing check on any head fit here,
whatever the dataset: a head reading generation traces must fall toward chance
under it, and one that does not has learned content.

`crop` is the fix, and it is a fix to preprocessing rather than to the head:
the traces live at the pixel scale, so the trunk has to be shown pixels at that
scale or the head has nothing forensic to fit. A 224 window of a 1024px image is
also ~5% of its area, which cuts how much scene semantics is on offer.

`multiscale` generalises `crop` along the axis `crop` fixed at one value. A
single 224 window still lets the model infer source resolution from how much
scene it can see; drawing the window size at random over 128-512px removes that
too, at the cost of the global artefacts a whole-image view carries -- layout
regularities, colour statistics, aspect ratio. The trade runs the way this
project needs, since generation traces are local and high-frequency, but it
means the 32x32 round-trip check above becomes the load-bearing test: a
multi-scale head that still survives it has learned content anyway.
"""

VIEWS = {
    "crop200": ("crop", 200),
    "resample512": ("resample", 512),
}
"""Eval-arm modes -> (geometry, size). See `preprocessing.degrade.crop`.

Sizes are in the name rather than a separate config key so that `input_mode`
alone identifies the protocol, which is what lets `_assert_head_matches` stay a
string comparison and what keeps a detector config self-describing.
"""

HEAD_COMPATIBILITY = {
    "resize": {"resize"},
    "crop": {"crop"},
    "multiscale": {"multiscale", "crop200", "resample512"},
}
"""Which `input_mode`s a head trained under a given mode may be scored under.

The guard this feeds exists to catch a head being handed a feature space it was
never fit on -- a resize-trained head shown native crops scores nonsense, and
does it silently. Plain equality expressed that correctly until training and
evaluation used different protocols on purpose: a multi-scale-trained head is
*meant* to be scored on both fixed arms, so equality would refuse the very
comparison the benchmark is built to make.

So the check is membership, not equality, and the permissive entry is exactly
one. `resize` and `crop` heads still refuse everything but themselves, which
keeps the original guarantee for every head trained before this existed.
Allowing `multiscale` -> `resample512` is deliberate even though that arm is out
of distribution: measuring the drop is the arm's purpose.
"""

POOLS = ("cls", "patchmean", "cls+patchmean")
"""How the token sequence becomes one vector.

DINOv3 emits `[CLS, reg_0..reg_3, patch_0..patch_N]`. The registers are dropped:
they are scratch space the model uses to park high-norm activations, deliberately
not tied to image content.

    cls            (D,)    the global descriptor, and only that
    patchmean      (D,)    mean over patch tokens -- local texture, averaged
    cls+patchmean  (2D,)   both, concatenated

`cls+patchmean` is DINOv3's own linear-probe recipe and the default here. It also
happens to be the right choice for this project specifically: generation traces
are a local high-frequency phenomenon, so a detector reading only the CLS token
would lean on semantics and degrade for reasons that have nothing to do with the
artefacts JPEG and blur destroy.
"""


def _pool_width(pool: str, hidden: int) -> int:
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    return hidden * (2 if pool == "cls+patchmean" else 1)


class ProbeHead(nn.Module):
    """Pooled features -> one logit. Fake-positive, like every `forward` here.

    Deliberately plain. The head is the thing GRACE holds frozen and
    differentiates through, and every extra inductive bias in it is one more
    explanation for a retention number that is not "the adapter did it". An MLP
    rather than a bare linear layer only because `train.weighting` claims
    one Jacobian implementation covers both cases, and a linear head would let
    that claim go untested on a real model.

    Gradient must flow to the *input*: the decision-weighted objective takes
    ∇_f head(f) at the clean features. Parameters are frozen by
    `FrozenDetector.freeze`; the input's graph is untouched by that.
    """

    def __init__(self, dim: int, hidden: int = 512, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        layers: list[nn.Module] = [nn.LayerNorm(dim)]
        width = dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f).squeeze(-1)


class DINOv3MLPDetector(FrozenDetector):
    """Frozen DINOv3 ViT-S/16 trunk + a trained MLP probe head.

    Parameters
    ----------
    backbone_id     : Hub id or local path for the DINOv3 ViT
    head_checkpoint : weights written by `train_probe.py`. `None` builds a
                      RANDOM head -- valid only for rendering a cache or for
                      tests, never for a reported number, and `head_untrained`
                      is set so callers can refuse it.
    pool            : see POOLS
    hidden, n_layers, dropout : head geometry. Ignored when a checkpoint is
                      given, which carries its own -- a checkpoint that
                      disagreed with the config would otherwise load into the
                      wrong shape or, worse, the right shape by accident.
    """

    def __init__(
        self,
        backbone_id: str = DEFAULT_BACKBONE,
        head_checkpoint: str | None = None,
        pool: str = "cls+patchmean",
        input_mode: str = "resize",
        hidden: int = 512,
        n_layers: int = 2,
        dropout: float = 0.0,
        revision: str | None = None,
        name: str | None = None,
    ):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")

        self.backbone_id = backbone_id
        self.name = name or "dinov3-vits16"
        self.pool = pool
        self.input_mode = input_mode
        self.processor = AutoImageProcessor.from_pretrained(backbone_id, revision=revision)
        self.backbone = AutoModel.from_pretrained(backbone_id, revision=revision)

        cfg = self.backbone.config
        self.n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0))
        self.feature_dim = _pool_width(pool, cfg.hidden_size)
        self.input_size = _input_size(self.processor)

        payload = None
        if head_checkpoint is not None:
            payload = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
            _assert_head_matches(
                payload, head_checkpoint, backbone_id, pool, self.feature_dim, input_mode
            )
            hidden = payload["hidden"]
            n_layers = payload["n_layers"]
            dropout = payload.get("dropout", 0.0)

        self.head_module = ProbeHead(self.feature_dim, hidden, n_layers, dropout)
        self.head_untrained = payload is None
        if payload is not None:
            self.head_module.load_state_dict(payload["state_dict"])

        self._preprocess = _build_preprocess(self.processor, input_mode, self.input_size)
        self.freeze()

    # -- the seam ------------------------------------------------------------
    # `trunk` and `head` are public API here rather than an afterthought:
    # `eval.splits.dinov3.DINOv3Split` is a two-line delegation to them, which
    # is the entire point of building this detector.

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, feature_dim). Registers dropped; see POOLS."""
        return self.pool_tokens(self.backbone(pixel_values=x).last_hidden_state)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, hidden) -> (B, feature_dim). Registers dropped; see POOLS.

        Split out of `trunk` so the ladder can pool an *intermediate* block's
        tokens through the identical code -- see
        `eval.splits.dinov3.DINOv3Split.trunk_with_taps`. Sharing the function
        rather than restating three lines is what makes a tap at the last block
        reduce to the seam feature exactly, which is the check `verify_taps`
        runs and the reason a tapped forward cannot silently drift from the
        features every baseline was measured at.
        """
        cls = tokens[:, 0]
        patches = tokens[:, self.n_prefix :].mean(dim=1)
        if self.pool == "cls":
            return cls
        if self.pool == "patchmean":
            return patches
        return torch.cat([cls, patches], dim=-1)

    def head(self, f: torch.Tensor) -> torch.Tensor:
        return self.head_module(f)

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._preprocess(img)

    def preprocess_fn(self):
        return self._preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def _build_preprocess(processor, input_mode: str, input_size: int):
    """The one place `input_mode` becomes a transform.

    `multiscale` resolves to the plain processor transform on purpose: under
    that protocol the window was already chosen upstream, in the dataset, where
    it can be seeded on the image index. Preprocessing must stay deterministic
    or `sha_preprocess` refuses it and the feature cache loses its meaning.
    """
    if input_mode == "crop":
        return _CropPreprocess(processor, input_size)
    if input_mode in VIEWS:
        geometry, size = VIEWS[input_mode]
        cls = _CropResizePreprocess if geometry == "crop" else _ResamplePreprocess
        return cls(processor, size)
    return _ProcessorPreprocess(processor)


def _input_size(processor) -> int:
    """Square side the processor resizes to -- `verify_split` builds its probe
    from this, and a 224 default would silently pass on a 512-input mirror."""
    size = getattr(processor, "crop_size", None) or getattr(processor, "size", None) or {}
    if not isinstance(size, dict):
        size = {k: getattr(size, k, None) for k in ("height", "width", "shortest_edge")}
    for key in ("height", "shortest_edge", "width"):
        if size.get(key):
            return int(size[key])
    return 224


def _assert_head_matches(payload, path, backbone_id, pool, feature_dim, input_mode) -> None:
    """A head trained on other features is the silent failure worth guarding.

    Every mismatch here produces a model that loads, runs, and scores nonsense: a
    `cls`-pooled head on `cls+patchmean` features fails loudly on the shape, but
    a head trained against a *different backbone* of the same width does not, and
    neither does one trained on crops and handed resized images.
    """
    # `input_mode` predates no checkpoint -- it was added after the first heads
    # were trained, and `resize` is the only thing those can have been fit on.
    # Defaulting to it (rather than skipping the check, as the optional keys
    # below do) is what makes an old head refuse a `crop` detector.
    stored_mode = payload.get("input_mode", "resize")
    allowed = HEAD_COMPATIBILITY.get(stored_mode, {stored_mode})
    if input_mode not in allowed:
        extra = (
            ""
            if len(allowed) == 1
            else f" That head may be scored under {sorted(allowed)}."
        )
        raise ValueError(
            f"{path} was trained on input_mode={stored_mode!r} but this detector "
            f"feeds input_mode={input_mode!r}. The trunk sees a different image "
            f"scale in each mode, so the features are not the same space.{extra} "
            f"Retrain the probe, or fix `input_mode` in the detector config."
        )
    stored_pool = payload.get("pool")
    if stored_pool is not None and stored_pool != pool:
        raise ValueError(
            f"{path} was trained on pool={stored_pool!r} but this detector emits "
            f"pool={pool!r}. Retrain the probe, or fix `pool` in the detector config."
        )
    stored_dim = payload.get("feature_dim")
    if stored_dim is not None and int(stored_dim) != feature_dim:
        raise ValueError(
            f"{path} was trained on {stored_dim}-d features; this detector emits "
            f"{feature_dim}-d."
        )
    stored_backbone = payload.get("backbone_id")
    if stored_backbone is not None and stored_backbone != backbone_id:
        raise ValueError(
            f"{path} was trained on backbone {stored_backbone!r} but this detector "
            f"loads {backbone_id!r}. Same width, different weights -- the head would "
            f"score a feature space it has never seen. Retrain the probe."
        )
