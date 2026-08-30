"""Patch-DCT extraction: the frequency branch's read of the image.

`dct` is the arithmetic (numpy only, pure, fingerprintable); `view` is the
picklable callable the render and evaluation paths hold. Importing `view` here
pulls torch in, which `dct` alone does not need -- `analyze_freq.py` and the
audit scripts import `pipeline.freq.dct` directly for that reason.
"""

from pipeline.freq.dct import (
    DEFAULT_BANDS, DEFAULT_GRID, DEFAULT_PATCH,
    band_masks, cell_pool, dct_basis, extract_freq, freq_fingerprint,
    normalize, patch_dct, radial_order,
)
from pipeline.freq.view import GEOMETRIES, FreqExtract

__all__ = [
    "DEFAULT_BANDS", "DEFAULT_GRID", "DEFAULT_PATCH", "GEOMETRIES",
    "FreqExtract", "band_masks", "cell_pool", "dct_basis", "extract_freq",
    "freq_fingerprint", "normalize", "patch_dct", "radial_order",
]
