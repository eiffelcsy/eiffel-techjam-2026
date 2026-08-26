"""Picklable preprocessing callables shared by the zoo adapters.

`FrozenDetector.preprocess_fn` exists because the Dataset is forked into
DataLoader workers: a bound method would drag every model parameter along and
fail outright once the model is on CUDA or MPS. Everything here is a
module-level class holding nothing but constants, so it pickles cleanly -- which
rules out `transforms.Lambda`, the usual way to express TenCrop stacking, since
a lambda does not pickle at all.

The three detectors disagree about almost everything -- B-Free normalizes at
native resolution, GAPL center-crops with ImageNet statistics despite a CLIP
backbone, RINE center-crops with CLIP statistics -- so the differences live in
the constructor arguments and the pipeline stays model-agnostic.
"""

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_STATS = (
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)

NORM_STATS: dict[str, tuple[tuple, tuple]] = {
    # B-Free names its normalization in the config.yaml shipped with the
    # weights; these are the four its `get_list_norm` understands.
    "resnet": IMAGENET_STATS,
    "clip": CLIP_STATS,
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "xception": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float CHW in [0, 1], the `ToTensor()` half of every recipe."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class Normalize:
    """ToTensor + Normalize, at whatever resolution the image arrives in.

    B-Free's entire inference transform. It resizes nothing: the 504x504
    cropping happens later, inside the network, in token space.
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return (_to_tensor(img) - self.mean) / self.std


class CenterCrop(Normalize):
    """CenterCrop to `size`, then normalize. No resize -- that is the protocol.

    GAPL and RINE both crop rather than resize, following the CNNDetection
    lineage: rescaling an image resamples away the very high-frequency traces
    these detectors read.

    Refuses images smaller than `size` instead of letting torchvision zero-pad
    them. A black border is a strong, label-uncorrelated artefact, and a
    detector scored on it is measuring the padding, not the image. None of the
    eleven degradations can trigger this -- every one returns an image the same
    size as it got -- so it only fires on a genuinely small source image, where
    silently proceeding would quietly invalidate the run.
    """

    def __init__(self, size: int, mean, std):
        super().__init__(mean, std)
        self.size = int(size)

    def _check(self, img: Image.Image) -> None:
        if min(img.size) < self.size:
            raise ValueError(
                f"image is {img.size[0]}x{img.size[1]}, smaller than the {self.size}px "
                "center crop this detector takes. CenterCrop would zero-pad it, and the "
                "detector would score the black border. Filter the manifest to images of "
                f"at least {self.size}px on both sides."
            )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        self._check(img)
        return super().__call__(F.center_crop(img, [self.size, self.size]))


class TenCrop(CenterCrop):
    """Four corners, the center, and the horizontal flip of each: (10, C, H, W).

    RINE's high-resolution protocol. `collate` stacks these into
    (B, 10, C, H, W), which the detector's `forward` flattens, scores, and means
    back down -- averaging *logits*, before the sigmoid, as upstream does.
    """

    def __call__(self, img: Image.Image) -> torch.Tensor:
        self._check(img)
        crops = F.ten_crop(img, [self.size, self.size])
        return torch.stack([(_to_tensor(c) - self.mean) / self.std for c in crops])
