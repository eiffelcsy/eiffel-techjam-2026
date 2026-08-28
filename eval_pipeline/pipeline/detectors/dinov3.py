"""DINOv3 ViT-S/16 + a linear-probe MLP head -- the detector GRACE's PoC adapts.

Every other detector in this package is somebody else's published model, loaded
from somebody else's checkpoint. This one is built here, and it exists for one
reason: **GRACE needs a detector whose trunk/head seam is not in dispute.**

The zoo splits (`grace.splits.rine`, `.bfree`, `.gapl`) each reconstruct a seam
inside a repo cloned by hand under `third_party/`, and `RINESplit._head_forward`
is explicitly marked unverified against its clone. Until those clones exist,
nothing in the GRACE pipeline can be run end to end on a real detector -- which
means the interesting claims (does the clean teacher buy retention, does the
adapter erase forensic evidence) cannot be tested at all.

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
across probe retrainings -- `grace.cache.spec` hashes the detector's *config*,
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

from pipeline.detectors.base import FrozenDetector
from pipeline.detectors.hf import _CropPreprocess, _ProcessorPreprocess

DEFAULT_BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"

INPUT_MODES = ("resize", "crop")
"""How a source image becomes the model's 224x224 input.

    resize   the processor's own transform: squash the whole image to 224x224
    crop     center 224x224 window at the source's native resolution

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
    rather than a bare linear layer only because `grace.train.weighting` claims
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

        self._preprocess = (
            _CropPreprocess(self.processor, self.input_size)
            if input_mode == "crop"
            else _ProcessorPreprocess(self.processor)
        )
        self.freeze()

    # -- the seam ------------------------------------------------------------
    # `trunk` and `head` are public API here rather than an afterthought:
    # `grace.splits.dinov3.DINOv3Split` is a two-line delegation to them, which
    # is the entire point of building this detector.

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, feature_dim). Registers dropped; see POOLS."""
        return self.pool_tokens(self.backbone(pixel_values=x).last_hidden_state)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, hidden) -> (B, feature_dim). Registers dropped; see POOLS.

        Split out of `trunk` so the ladder can pool an *intermediate* block's
        tokens through the identical code -- see
        `grace.splits.dinov3.DINOv3Split.trunk_with_taps`. Sharing the function
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
    if stored_mode != input_mode:
        raise ValueError(
            f"{path} was trained on input_mode={stored_mode!r} but this detector "
            f"feeds input_mode={input_mode!r}. The trunk sees a different image "
            f"scale in each mode, so the features are not the same space. Retrain "
            f"the probe, or fix `input_mode` in the detector config."
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
