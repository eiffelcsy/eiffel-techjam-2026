"""The frequency side-view: `PIL image -> (cells, coeffs)` tensor, picklable.

`dct.py` is deliberately torch-free and pure -- it is the arithmetic, and the
falsification script, the renderer and the detector all have to run the identical
arithmetic or the cached view is a different feature from the one scored at test
time. This module is the thin callable wrapper the three paths actually hold:

    render   train.cache.writer.MultiViewDataset, in a DataLoader worker
    train    train.loop.train_enrich, through the cached view
    eval     freq_branch.detectors.fused.FusedDetector.aux_fn(), in a worker again

Module-level class, holding five ints and two strings, for the same reason
`degrade.crop.SampleCrop` and `detectors.hf._ProcessorPreprocess` are: it is
handed to a Dataset that gets forked (or pickled, under `spawn`) into workers,
and anything it closed over would travel with it.

WHY IT CARRIES A GEOMETRY. The spatial branch and the frequency branch must read
the *same window*, and the two paths choose that window in different places. On
the training path the window is already chosen upstream, in the dataset, by
`multiscale_crop` -- so `geometry` is empty and this reads whatever it is handed.
On the evaluation arms the window is chosen inside the detector's own
preprocessing (`_CropResizePreprocess`, `_ResamplePreprocess`), which returns a
normalized 224px tensor this cannot read backwards from. So the arm's geometry is
restated here, from the same `VIEWS` table the preprocessing reads, and applied
to the PIL image before the DCT. Restating it is the cost of the two branches
consuming the image at different points; `tests/test_freq_view.py` pins the two
to the same window.
"""

import numpy as np
import torch
from PIL import Image

from preprocessing.degrade.crop import fixed_crop, fixed_resample
from freq_branch.dct import (
    DEFAULT_GRID, DEFAULT_PATCH, extract_freq, freq_fingerprint,
)

GEOMETRIES = ("", "crop", "resample")
"""Window selection applied before the DCT.

    ""          none -- the caller already chose the window (training/render)
    "crop"      centre `size` window at native pixel scale (eval arm (a))
    "resample"  whole image squashed to `size` x `size`  (eval arm (b))
"""

SOURCES = ("window", "native")
"""The render path's window selection, which pixels the DCT reads.

    "window"  the same cropped window the spatial branch reads (the status quo)
    "native"  the whole degraded image at native resolution, BEFORE the crop

`"native"` exists to test the complementarity premise at its strongest: a full
native-res DCT sees frequencies and scene content the 128-256px window the trunk
reads threw away. The cost is that the 196 cells now tile the whole image rather
than the window, so the spatial and frequency branches are no longer aligned
cell-for-cell -- the enricher's cross-attention must learn the correspondence
itself. That ambiguity is the trade the arm exists to measure, and `source` is
fingerprinted so the two renderings can never be read interchangeably.
"""


class FreqExtract:
    """`(PIL image) -> (grid*grid, channels*patch**2)` float32 tensor.

    Output shape is fixed by `(grid, patch, channels)` alone, independent of the
    input size: that is what lets one cached frequency view serve crops drawn
    anywhere in the 128-512px range, and what lets the tokens stack in the
    harness `collate` with no special handling.
    """

    def __init__(
        self,
        patch: int = DEFAULT_PATCH,
        grid: int = DEFAULT_GRID,
        radial: bool = True,
        geometry: str = "",
        size: int = 0,
        source: str = "window",
    ):
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        if geometry and size < patch:
            raise ValueError(
                f"geometry={geometry!r} needs a size of at least one {patch}x{patch} "
                f"block, got {size}"
            )
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        self.patch, self.grid, self.radial = int(patch), int(grid), bool(radial)
        self.geometry, self.size = str(geometry), int(size)
        self.source = str(source)

    def window(self, img: Image.Image) -> Image.Image:
        """The pixels the DCT will read. Public so a test can compare it against
        what the spatial branch's preprocessing selects."""
        if self.geometry == "crop":
            return fixed_crop(img, self.size)
        if self.geometry == "resample":
            return fixed_resample(img, self.size)
        return img

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(self.window(img).convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(
            np.ascontiguousarray(extract_freq(arr, self.patch, self.grid, self.radial))
        )

    def shape(self, channels: int = 3) -> tuple[int, int]:
        return (self.grid * self.grid, channels * self.patch * self.patch)

    def fingerprint(self, channels: int = 3, norm: str = "log1p") -> str:
        """`CacheSpec.freq_sha`. The GEOMETRY is deliberately not in it.

        The coefficient set is a render-time commitment that a cache cannot be
        read against if it changes; the geometry is a property of the *arm*, and
        arm (a) and arm (b) are meant to read the same rendered protocol through
        different windows. Putting the window in here would make the two arms
        mutually unreadable for no gain -- the window is already covered by
        `crop_sha` on the render path and by `input_mode` on the eval path.

        `source` IS in it, and that is the one deliberate departure: unlike the
        eval-arm geometry, which two arms apply over one rendered view, the
        render-time source decides WHICH pixels were rendered. A `"native"`
        cache holds DCTs of whole images and a `"window"` cache holds DCTs of
        crops; the two are different features of different pictures and must
        never be read interchangeably.
        """
        return freq_fingerprint(
            self.patch, self.grid, channels, self.radial, norm, source=self.source
        )
