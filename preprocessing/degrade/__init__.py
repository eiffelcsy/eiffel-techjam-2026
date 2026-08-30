"""The eleven transforms, the four composition levels, and the crop that
selects which window of the degraded image the model actually sees."""

from preprocessing.degrade.conditions import (
    LEVELS, Condition, Recipe, Step, build_conditions, load_grid,
)
from preprocessing.degrade.crop import (
    POLICIES, SAMPLE_EPOCH, CropDraw, SampleCrop, crop_fingerprint, draw_size,
    fixed_crop, fixed_resample, multiscale_crop,
)
from preprocessing.degrade.ops import TRANSFORMS, Transform, register

__all__ = [
    "LEVELS", "Condition", "Recipe", "Step", "build_conditions", "load_grid",
    "POLICIES", "SAMPLE_EPOCH", "CropDraw", "SampleCrop", "crop_fingerprint",
    "draw_size", "fixed_crop", "fixed_resample", "multiscale_crop",
    "TRANSFORMS", "Transform", "register",
]
