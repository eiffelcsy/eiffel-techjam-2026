"""GAPL -- scaling up AI-generated image detection with generator-aware prototypes.

Qin et al. https://github.com/UltraCapture/GAPL

A CLIP ViT-L/14 vision tower with LoRA adapters projects an image to a
128-d forensic feature, which cross-attends over 64 learned generator-aware
prototypes before a linear head. The prototypes are what make it scale: rather
than one decision boundary per generator family, the model learns a bank of
generator signatures and asks how strongly an image matches any of them.

Two upstream details are worked around here rather than patched into the clone,
since an edited clone is silently lost the next time it is pulled:

  * `models.py` loads CLIP from a hardcoded absolute snapshot path under
    /root/.cache with local_files_only=True, in a call that fires inside
    __init__ before we can reach the object. `_RedirectCLIP` intercepts that one
    call for the duration of construction.
  * `GAPLModel.__init__` defaults to device='cuda' and moves itself there
    immediately, so it cannot be built on a CPU box at all. It is always built
    on CPU here; `build_detector` moves it afterwards.

And one that is a correctness trap rather than an inconvenience: upstream loads
the checkpoint with `strict=False`, which will happily leave a randomly
initialised head in place. That failure looks exactly like a weak detector --
an AUC near 0.5 -- rather than like an error, so the key sets are checked here
and a mismatch raises.

Output is a fake-positive logit thresholded at zero (upstream applies sigmoid
and thresholds at 0.5, which is the same boundary), so it maps onto the contract
unchanged.

Set up with:
    git clone https://github.com/UltraCapture/GAPL.git third_party/GAPL
    hf download AbyssLumine/GAPL checkpoint.pt --local-dir checkpoints/gapl
"""

import torch
from PIL import Image

from pipeline.detectors._preprocess import IMAGENET_STATS, CenterCrop
from pipeline.detectors._vendor import require_file, vendored
from pipeline.detectors.base import FrozenDetector

CLIP_ID = "openai/clip-vit-large-patch14"
FROZEN_PREFIX = "feature_extractor."
"""Keys the checkpoint legitimately omits: the CLIP tower comes from the Hub,
not from the checkpoint, so only its LoRA deltas are saved. A missing key
*outside* this prefix is an untrained head."""


class _RedirectCLIP:
    """Point GAPL's hardcoded CLIP path at a Hub id, for one construction.

    Restores the original on exit so nothing else in the process sees a patched
    `transformers`, and so a failed construction does not leave the patch behind.
    """

    def __init__(self, clip_id: str):
        self.clip_id = clip_id

    def __enter__(self):
        import transformers

        self._original = transformers.CLIPModel.from_pretrained

        def patched(_path, *args, **kwargs):
            # local_files_only was paired with the absolute path; dropping it
            # lets the Hub id resolve through the normal cache.
            kwargs.pop("local_files_only", None)
            return self._original(self.clip_id, *args, **kwargs)

        transformers.CLIPModel.from_pretrained = patched
        return self

    def __exit__(self, *exc):
        import transformers

        transformers.CLIPModel.from_pretrained = self._original


class GAPL(FrozenDetector):
    """GAPL as a frozen detector.

    Parameters
    ----------
    checkpoint : stage-2 checkpoint holding both `model` and `prototype`
    clip_id    : Hub id for the CLIP tower, replacing the hardcoded local path
    input_size : center-crop size; 224 is the trained protocol
    name       : display name in results
    """

    def __init__(
        self,
        checkpoint: str = "checkpoints/gapl/checkpoint.pt",
        clip_id: str = CLIP_ID,
        input_size: int = 224,
        name: str = "gapl",
    ):
        super().__init__()
        with vendored("GAPL"):
            import models

        path = require_file(
            checkpoint,
            "GAPL checkpoint",
            "Download it with: hf download AbyssLumine/GAPL checkpoint.pt "
            "--local-dir checkpoints/gapl",
        )
        payload = torch.load(path, map_location="cpu")

        with _RedirectCLIP(clip_id):
            # device='cpu' is not a preference: the default 'cuda' is applied
            # inside __init__ and fails outright without a GPU.
            self.model = models.GAPLModel(
                fe_path=None, proto_path=None, freeze_backbone=False, device="cpu"
            )

        # strict=False is correct here -- the CLIP tower is not in the
        # checkpoint -- but only alongside the check below.
        missing, unexpected = self.model.load_state_dict(payload["model"], strict=False)
        untrained = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
        if untrained or unexpected:
            raise RuntimeError(
                f"{path} does not match GAPLModel: "
                f"{len(untrained)} head parameter(s) left uninitialised "
                f"{untrained[:5]}, {len(unexpected)} unexpected {list(unexpected)[:5]}. "
                "Loading this would score a random head and look like a weak detector."
            )
        self.model.load_prototype(payload["prototype"])

        self.name = name
        self.clip_id = clip_id
        self._preprocess = CenterCrop(input_size, *IMAGENET_STATS)
        self.freeze()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._preprocess(img)

    def preprocess_fn(self):
        return self._preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # The head is Linear(128 -> 1), so this is (B, 1); tolerate (B,) too.
        return out.squeeze(-1) if out.ndim == 2 else out
