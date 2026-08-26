"""RINE -- representations from intermediate encoder-blocks.

Koutlis & Papadopoulos, ECCV 2024. https://github.com/mever-team/rine

CLIP ViT-L/14 stays entirely frozen; forward hooks pull the CLS token out of all
24 transformer blocks, and a small trainable head learns which blocks matter via
a softmax-weighted sum (their Trainable Importance Estimator). The insight is
that the low-level generation traces live in the early blocks, which the final
CLIP embedding has already discarded.

Because only the head is trained, the released checkpoints are tiny -- 1 to
40 MB -- and ship inside the repo, with the ~890 MB CLIP downloaded separately
by `clip.load` on first use.

Four checkpoints exist, differing in what they were trained on and in the head
geometry that goes with it (see NCLS_CONFIG). `ldm` is the latent-diffusion one
and the right default for a diffusion-era eval set; the `Nclass` ones follow the
ProGAN/ForenSynths protocol and make a useful contrast, so both are configured.

Two upstream loading details are deliberately not reused:

  * `get_our_trained_model` resolves the checkpoint against a relative `ckpt/`
    path, so it only works with the cwd inside their repo.
  * It assigns weights through a string-manipulating `exec()`, one parameter at
    a time. Assignment cannot fail the way `load_state_dict` can: a name that
    does not match simply creates a new attribute, and the model scores on
    randomly initialised weights with no error. A real `load_state_dict` plus
    the two key assertions below is the same operation, minus the silence.

Output is a fake-positive logit thresholded at zero (upstream sigmoids and
thresholds at 0.5, the same boundary), so it maps onto the contract unchanged.

Set up with:
    git clone https://github.com/mever-team/rine.git third_party/rine
    pip install "robust-aigc-eval[zoo]"    # brings in openai/CLIP
"""

import torch
from PIL import Image

from pipeline.detectors._preprocess import CLIP_STATS, CenterCrop, TenCrop
from pipeline.detectors._vendor import require_file, vendored
from pipeline.detectors.base import FrozenDetector

NCLS_CONFIG: dict[object, tuple[int, int]] = {
    # training setting -> (nproj, proj_dim); the head geometry is not a free
    # choice, it has to match the checkpoint that was trained with it
    1: (4, 1024),
    2: (4, 128),
    4: (2, 1024),
    "ldm": (4, 1024),
}
FROZEN_PREFIX = "clip."
"""The checkpoints are saved as trainable parameters only -- CLIP keys are
stripped at save time -- so every missing key must be under this prefix."""


class RINE(FrozenDetector):
    """RINE as a frozen detector.

    Parameters
    ----------
    checkpoint : path to one of their `model_*_trainable.pth` files
    ncls       : which training setting the checkpoint is; selects the head geometry
    tencrop    : ten-crop aggregation (their high-resolution protocol) instead of
                 a single center crop
    input_size : crop size; 224 is the trained protocol
    name       : display name in results
    """

    def __init__(
        self,
        checkpoint: str = "third_party/rine/ckpt/model_ldm_trainable.pth",
        ncls: object = "ldm",
        tencrop: bool = False,
        input_size: int = 224,
        name: str = "rine",
    ):
        super().__init__()
        if ncls not in NCLS_CONFIG:
            raise ValueError(f"ncls must be one of {list(NCLS_CONFIG)}, got {ncls!r}")
        nproj, proj_dim = NCLS_CONFIG[ncls]

        with vendored("rine"):
            from src.models import Model

        path = require_file(
            checkpoint,
            f"RINE {ncls} checkpoint",
            "It ships inside the clone -- check third_party/rine/ckpt/.",
        )
        # device='cpu' here is load-time only; build_detector moves the model.
        # It also matters for dtype: see the .float() note below.
        self.model = Model(
            backbone=("ViT-L/14", 1024), nproj=nproj, proj_dim=proj_dim, device="cpu"
        )
        missing, unexpected = self.model.load_state_dict(
            torch.load(path, map_location="cpu"), strict=False
        )
        untrained = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
        if untrained or unexpected:
            raise RuntimeError(
                f"{path} does not match a RINE head with ncls={ncls!r} "
                f"(nproj={nproj}, proj_dim={proj_dim}): "
                f"{len(untrained)} head parameter(s) left uninitialised {untrained[:5]}, "
                f"{len(unexpected)} unexpected {list(unexpected)[:5]}. "
                "Most likely `ncls` does not match this checkpoint."
            )

        # OpenAI's clip.load keeps the model in fp16 unless it was loaded onto
        # CPU. Constructing on CPU already gives fp32, but this is asserted
        # rather than assumed: the mismatch would only appear once the model
        # reaches a GPU, long after this code was last looked at.
        self.model.clip.float()

        self.name = name
        self.ncls = ncls
        self.tencrop = tencrop
        crop = TenCrop if tencrop else CenterCrop
        self._preprocess = crop(input_size, *CLIP_STATS)
        self.freeze()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._preprocess(img)

    def preprocess_fn(self):
        return self._preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The model returns (logit, projected feature); the feature exists for
        # the training-time contrastive loss and is discarded here.
        if not self.tencrop:
            return self.model(x)[0].squeeze(-1)

        # (B, 10, C, H, W): score every crop, then mean the *logits*, before the
        # sigmoid, as upstream's evaluation does. Averaging probabilities
        # instead would compress the scores toward 0.5 and flatten the AUC.
        batch, crops = x.shape[:2]
        logits = self.model(x.reshape(-1, *x.shape[2:]))[0]
        return logits.reshape(batch, crops).mean(dim=1)
