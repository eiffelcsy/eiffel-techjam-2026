"""Dataset yielding (tensor, meta).

Format neutrality is the one correctness detail worth being paranoid about:
if reals are JPEGs and fakes are PNGs, a detector can learn the container
instead of the content. Every image goes through an identical decode path
before anything else touches it, and sources write a single on-disk format, so
no container statistic is left correlated with the label.
"""

from typing import NamedTuple

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset

from common.io import list_images


class Inputs(NamedTuple):
    """A model input that is more than one tensor. The exception, not the rule.

    Almost every detector in this harness takes a preprocessed image and nothing
    else, and that path is unchanged: with no auxiliary extractor the dataset
    yields a bare tensor, `collate` stacks it, and `detector.score(batch.to(
    device))` sees exactly what it always saw, byte for byte.

    This exists for one detector shape -- a second branch that must read the
    image in a basis preprocessing destroys. `grace.detectors.fused.FusedDetector`
    runs a patch-DCT over the same window at NATIVE pixel scale; by the time the
    224 tensor exists that information is gone, and it cannot be recovered from
    the tensor at any cost. So the second read happens in the worker, beside the
    first, and travels with it.

    `.to()` is the whole interface the runner needs, because the runner's one
    line is `detector.score(batch.to(device))`.
    """

    x: torch.Tensor
    aux: torch.Tensor

    def to(self, *args, **kwargs) -> "Inputs":
        return Inputs(self.x.to(*args, **kwargs), self.aux.to(*args, **kwargs))

# PIL caps how far it will inflate a PNG's ancillary chunks (1MB by default) as
# a zlib-bomb guard, and raises on anything larger -- an ICC profile fatter than
# the image is rare but does occur upstream. We decode local dataset
# directories, not untrusted uploads, and the profile is discarded a line later
# anyway, so the guard only costs us images. Raised, not removed.
PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024


def load_normalized(path: str) -> Image.Image:
    """Decode to RGB and drop everything that is not pixels.

    EXIF, ICC profiles and the format tag never reach the detector: `convert`
    returns a fresh image whose pixels are its own, and clearing `info` and
    `format` drops the metadata that came along with it. There is no re-encode
    step: PNG is lossless, so a round-trip would return the identical array, and
    it cannot undo compression artefacts already baked into the pixels of a
    lossy source. Uniform on-disk format is the source's job.

    This used to rebuild the image from `rgb.tobytes()`, which achieved the same
    guarantee by copying every pixel a second time -- 5.1 ms of the 12.5 ms this
    function cost per NTIRE image, or ~40%, to strip four JFIF keys. The cache
    render calls it 277,643 times.
    """
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
    rgb.info.clear()
    rgb.format = None
    return rgb


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class AIGCDataset(Dataset):
    """Rows of the manifest -> (preprocessed tensor, meta dict).

    One dataset per condition: the same manifest is wrapped once per
    (transform, parameter) pair, so scores across conditions line up row for
    row and are directly comparable. meta carries index, label, generator and
    the condition id.
    """

    def __init__(self, manifest, preprocess=None, condition=None, seed: int = 0,
                 crop=None, aux=None):
        self.paths = manifest["path"].tolist()
        self.labels = manifest["label"].tolist()
        self.generators = manifest["generator"].tolist()
        # The manifest index, not the row position: it is the stable image
        # identity that seeds degradations, so it must survive subsetting.
        self.index = manifest.index.tolist()
        self.preprocess = preprocess or _to_tensor
        self.condition = condition
        self.seed = seed
        self.crop = crop
        """Optional `(image, index) -> image` applied after the condition and
        before preprocessing -- `preprocessing.degrade.crop.SampleCrop`.

        Used by stage 0, so the probe head is fit on the same windows the cache
        renders. The evaluation sweep leaves it None and selects its window
        through the detector's `input_mode` instead, because an eval window must
        be identical across all 26 conditions: the retention denominator compares
        one image clean against itself degraded, and a per-condition window would
        quietly make that a comparison between two different pictures.
        """
        self.aux = aux
        """Optional `(image) -> tensor`, run on the same degraded image the
        preprocessing sees, from `detector.aux_fn()`.

        None for every detector in this harness but the fused one -- see
        `Inputs`. It runs HERE, in the worker, rather than in the model, because
        what it reads (native pixel scale) does not survive preprocessing, and
        because eight workers doing DCTs while the GPU runs the trunk is free
        where a serial pass in the parent is not.
        """

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        index = self.index[i]
        img = load_normalized(self.paths[i])
        recipe_label, transforms = "clean", ()
        if self.condition is not None:
            img, recipe = self.condition(img, index)
            recipe_label, transforms = recipe.label(), recipe.transforms()
        # Degrade, then crop: the recipe keeps acting at native resolution, so
        # the parameter grid keeps the calibration it was measured at.
        if self.crop is not None:
            img = self.crop(img, index)
        meta = {
            "index": int(index),
            "label": int(self.labels[i]),
            "generator": self.generators[i],
            "condition": getattr(self.condition, "id", "clean"),
            "recipe": recipe_label,
            # transform names, not the full label: the recipe tables group by
            # which transforms co-occurred, independent of their parameters.
            "transforms": transforms,
        }
        if self.aux is None:
            return self.preprocess(img), meta
        # Both branches read `img` -- the same degraded pixels, before the
        # 224 squash. That the two see one window is the invariant the whole
        # frequency branch rests on, and it is enforced by there being one
        # variable here rather than by anything asserting it.
        return Inputs(self.preprocess(img), self.aux(img)), meta


class ImageFolderDataset(Dataset):
    """Bare directory of images, for inference. No labels, no manifest."""

    def __init__(self, root: str, preprocess=None):
        self.paths = list_images(root)
        if not self.paths:
            raise ValueError(f"no images found under {root!r}")
        self.preprocess = preprocess or _to_tensor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_normalized(self.paths[i])
        return self.preprocess(img), {"image_path": str(self.paths[i])}


def collate(batch):
    """Stack tensors, keep metas as a plain list of dicts.

    The default collate would transpose the meta dicts into dicts of batched
    columns; the runner wants them per image.

    An `Inputs` item stacks both of its tensors and stays an `Inputs`. The
    auxiliary tensor is fixed-shape by construction -- `extract_freq` emits
    `(196, 192)` whatever the window size -- so it stacks with no special
    handling, which is a property of the extractor rather than luck.
    """
    tensors, metas = zip(*batch)
    if isinstance(tensors[0], Inputs):
        stacked = Inputs(
            torch.stack([t.x for t in tensors]),
            torch.stack([t.aux for t in tensors]),
        )
    else:
        stacked = torch.stack(tensors)
    return stacked, list(metas)
