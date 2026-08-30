"""Turning a manifest row into model input: degrade, window, normalize.

    preprocessing/dataset.py     AIGCDataset / ImageFolderDataset -- the
                                 torch Datasets that apply the below per row
    preprocessing/degrade/       the eleven transforms, the four composition
                                 levels, and the crop that selects which window
                                 of the degraded image the model actually sees
"""

from preprocessing.dataset import (
    AIGCDataset, ImageFolderDataset, Inputs, collate, load_normalized,
)

__all__ = [
    "AIGCDataset", "ImageFolderDataset", "Inputs", "collate", "load_normalized",
]
