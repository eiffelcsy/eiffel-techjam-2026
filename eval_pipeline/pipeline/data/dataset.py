"""Dataset yielding (tensor, meta).

Format neutrality is the one correctness detail worth being paranoid about:
if reals are JPEGs and fakes are PNGs, a detector can learn the container
instead of the content. Every image goes through an identical decode path
before anything else touches it, and sources write a single on-disk format, so
no container statistic is left correlated with the label.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pipeline.utils.io import list_images


def load_normalized(path: str) -> Image.Image:
    """Decode to RGB and drop everything that is not pixels.

    Rebuilding from raw bytes discards EXIF, ICC profiles and the format tag,
    so nothing about how the file was stored reaches the detector. There is no
    re-encode step: PNG is lossless, so a round-trip would return the identical
    array, and it cannot undo compression artefacts already baked into the
    pixels of a lossy source. Uniform on-disk format is the source's job.
    """
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())


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

    def __init__(self, manifest, preprocess=None, condition=None, seed: int = 0):
        self.paths = manifest["path"].tolist()
        self.labels = manifest["label"].tolist()
        self.generators = manifest["generator"].tolist()
        # The manifest index, not the row position: it is the stable image
        # identity that seeds degradations, so it must survive subsetting.
        self.index = manifest.index.tolist()
        self.preprocess = preprocess or _to_tensor
        self.condition = condition
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        index = self.index[i]
        img = load_normalized(self.paths[i])
        recipe_label, transforms = "clean", ()
        if self.condition is not None:
            img, recipe = self.condition(img, index)
            recipe_label, transforms = recipe.label(), recipe.transforms()
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
        return self.preprocess(img), meta


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
    """
    tensors, metas = zip(*batch)
    return torch.stack(tensors), list(metas)
