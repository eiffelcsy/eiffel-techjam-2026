"""B-Free -- a bias-free training paradigm for AI-generated image detection.

Guillaro et al., CVPR 2025. https://github.com/grip-unina/B-Free

A DINOv2-pretrained ViT-B/14 with four registers, trained on self-conditioned
inpainted images so the detector cannot latch onto the semantic or format biases
that separate a real dataset from a generated one. One released checkpoint,
`BFREE_dino2reg4`; there is no "dev" variant despite the name looking like one.

The unusual part is that it does not resize. The image goes into the network at
native resolution, the patch embedding is applied to the whole thing, and five
36x36 token windows (504px at stride 14) are cut in *token* space -- center and
four corners -- and their logits averaged. So the input tensor's size varies
per image, and a run of this detector must set `batch_size: 1`: `collate`
stacks, and only a batch of one stacks at heterogeneous sizes. Images below
504px are handled by the wrapper, which tiles the embedding to fill the window.

Its output needs no adaptation. B-Free thresholds a raw fake-positive logit at
zero, which is exactly what `FrozenDetector.forward` is defined to return, and
the paper reports NLL and ECE -- so `score`'s sigmoid is a calibrated P(fake)
rather than a monotone stand-in for one.

Licence note: B-Free's weights are released for nonprofit informational use
only (grip.unina.it/download/LICENSE_OPEN.txt), unlike RINE's Apache-2.0.

Set up with:
    git clone https://github.com/grip-unina/B-Free.git third_party/B-Free
    # unzip BFREE_dino2reg4.zip into checkpoints/bfree/
"""

from pathlib import Path

import torch
import yaml
from PIL import Image

from pipeline.detectors._preprocess import NORM_STATS, Normalize
from pipeline.detectors._vendor import require_file, vendored
from pipeline.detectors.base import FrozenDetector

WEIGHTS_URL = "https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip"


class BFree(FrozenDetector):
    """B-Free as a frozen detector.

    Parameters
    ----------
    weights_dir : directory holding the unzipped weight bundles
    model_name  : bundle to load; the only released one is BFREE_dino2reg4
    norm_type   : override the normalization named in the bundle's config.yaml
    name        : display name in results
    """

    def __init__(
        self,
        weights_dir: str = "checkpoints/bfree",
        model_name: str = "BFREE_dino2reg4",
        norm_type: str | None = None,
        name: str = "b-free",
    ):
        super().__init__()
        # Imported here, not at module scope, so `pipeline.detectors` stays
        # importable without the zoo cloned. See _vendor.vendored.
        with vendored("B-Free", "code"):
            from networks import get_network

        bundle = Path(weights_dir) / model_name
        hint = f"Download {WEIGHTS_URL} and unzip it into {Path(weights_dir).resolve()}."
        config_path = require_file(bundle / "config.yaml", f"{model_name} config", hint)
        with config_path.open() as f:
            config = yaml.safe_load(f)

        # Their own loader (`get_config`) resolves this against a hardcoded
        # relative ./weights directory, so it is reimplemented rather than
        # called: the path belongs in our config, not in their cwd.
        weights_path = require_file(
            bundle / config["weights_file"], f"{model_name} weights", hint
        )
        norm_type = norm_type or config["norm_type"]
        if norm_type not in NORM_STATS:
            raise ValueError(
                f"{config_path} names norm_type={norm_type!r}, which is not one of "
                f"{sorted(NORM_STATS)}. Pass args.norm_type to override."
            )

        self.name = name
        self.model_name = model_name
        self.arch = config["arch"]
        self.norm_type = norm_type
        self.model = get_network(self.arch)
        # strict: B-Free ships a complete checkpoint, so any key mismatch is a
        # real error. The other two zoo repos need strict=False for legitimate
        # reasons and pay for it with explicit key assertions instead.
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu")["model"])

        self._preprocess = Normalize(*NORM_STATS[norm_type])
        self.freeze()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._preprocess(img)

    def preprocess_fn(self):
        return self._preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Wrapper5crops has already split, scored and averaged the five token
        # windows, so this is one row per image. Two head widths exist upstream;
        # both reduce to the same fake-minus-real logit.
        out = self.model(x)
        return out[:, 0] if out.shape[1] == 1 else out[:, 1] - out[:, 0]
