"""The frequency branch: patch-DCT extraction, the cross-attention enricher,
and the detector that splices both onto a frozen GRACE adapter.

    dct.py               patch-DCT arithmetic (pure numpy, fingerprintable)
    view.py               FreqExtract -- the picklable (image) -> tensor wrapper
    models/frequency.py   FrequencyEnricher, BandExpert
    detectors/fused.py    FusedDetector (GRACE-freq)

Layered on top of `grace_adapter`, not a parallel branch: the enricher fuses
DCT tokens into the *adapter's* output, so `FusedDetector` freezes and loads a
`grace_adapter` checkpoint before anything here trains.

Importing this package pulls torch in via `view.py` -- `analyze_freq.py` and
the audit scripts that only need the arithmetic import `freq_branch.dct`
directly for that reason.
"""

from freq_branch.dct import (
    DEFAULT_BANDS, DEFAULT_GRID, DEFAULT_PATCH,
    band_masks, cell_pool, dct_basis, extract_freq, freq_fingerprint,
    normalize, patch_dct, radial_order,
)
from freq_branch.view import GEOMETRIES, FreqExtract

__all__ = [
    "DEFAULT_BANDS", "DEFAULT_GRID", "DEFAULT_PATCH", "GEOMETRIES",
    "FreqExtract", "band_masks", "cell_pool", "dct_basis", "extract_freq",
    "freq_fingerprint", "normalize", "patch_dct", "radial_order",
]
