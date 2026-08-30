"""Patch-DCT feature extraction: the frequency branch's read of the image.

    crop (s x s, native pixel scale)
      -> patch_dct        per 8x8 block, per channel, a 2D DCT-II
      -> |coefficients|   magnitudes, because phase does not survive pooling
      -> cell_pool        adaptive average to a fixed 14 x 14 cell grid
      -> normalize        log1p, so fp16 on disk keeps the small coefficients
      -> (196, 192)       196 cells, 3 channels x 64 radially-ordered coeffs

Pure functions over numpy arrays. No torch, no model state, no I/O: the render
path and the evaluation path must run the identical computation or the cached
frequency view is a different feature from the one the detector sees at test
time, and nothing downstream would catch it. Purity is also what makes the
output cache-fingerprintable and what lets `analyze_freq.py` (the E0 gate) use
the shipped extractor rather than a reimplementation of it.

Why 8x8. It is the JPEG block size, so the block-boundary artefacts the JPEG
degradation family produces land on coefficient positions this basis actually
resolves rather than smearing across it.

Why a fixed cell grid. The crop side varies over 128-512px, which is 16x16 to
64x64 blocks; pooling all of them to 14x14 gives the model a fixed-shape input
and one cell per DINOv3 patch token at 224. The cost is that a 512px crop
averages ~16x more blocks per cell than a 128px crop, so pooled statistics are
smoother at the coarse end of the scale range -- if that shows up as
scale-dependent variance in E0, the cell grid is the knob.

Why magnitudes. DCT coefficient signs are essentially arbitrary across blocks of
natural image content, so averaging signed coefficients over the blocks in a
cell cancels them toward zero and the feature dies. Cells pool |coefficient|.
"""

import numpy as np

DEFAULT_PATCH = 8
DEFAULT_GRID = 14
DEFAULT_BANDS = 2


def dct_basis(n: int = DEFAULT_PATCH) -> np.ndarray:
    """Orthonormal DCT-II basis, shape (n, n). `D @ block @ D.T` is the 2D DCT.

    Built here rather than imported from scipy so the extractor has no numeric
    dependency beyond numpy -- it runs inside DataLoader workers on the render
    path. `tests/test_freq_extraction.py` checks it against `scipy.fft.dctn`
    with `norm="ortho"`, which is the definition this reproduces.
    """
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    basis *= np.sqrt(2.0 / n)
    basis[0] /= np.sqrt(2.0)
    return basis


def radial_order(patch: int = DEFAULT_PATCH) -> np.ndarray:
    """Indices reordering a flattened patch by radial spatial frequency.

    Coefficient (u, v) sits at radius sqrt(u^2 + v^2); ties break on (u+v, u, v)
    so the order is total and stable across platforms. Applying this makes a
    frequency band a *contiguous slice* of the coefficient axis, which is what
    lets `band_masks` be clean rectangles and lets a top-k coefficient ablation
    (E13) mean "the k lowest frequencies" rather than "the k first in raster
    order". Index 0 is always DC.
    """
    u, v = np.meshgrid(np.arange(patch), np.arange(patch), indexing="ij")
    u, v = u.ravel(), v.ravel()
    radius = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
    return np.lexsort((v, u, u + v, radius))


def band_masks(
    patch: int = DEFAULT_PATCH,
    n_bands: int = DEFAULT_BANDS,
    channels: int = 1,
    soft: bool = False,
) -> np.ndarray:
    """Complementary masks over the radially-ordered coefficient axis.

    Returns `(n_bands, channels * patch**2)`. Bands partition the axis into
    equal-count contiguous slices -- band 0 is the lowest frequencies (DC
    first), band `n_bands-1` the highest. Masks sum to exactly 1 at every
    coefficient, so the bands are a decomposition of the spectrum rather than a
    selection from it: nothing the model could read is dropped by the split
    itself, only routed to one expert or the other.

    `soft` replaces the hard boundary with a raised-cosine crossfade one band
    wide, which keeps the sum-to-one property while removing the discontinuity
    at the split. Hard masks are the default: the HF/LF experts are meant to be
    able to specialise completely, and a hard edge in the *mask* is not an edge
    in the *image*.

    Masks tile across channels, matching `extract_freq`'s channel-major layout,
    so a band means the same frequencies in every colour channel.
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    n = patch * patch
    if n_bands > n:
        raise ValueError(f"n_bands={n_bands} exceeds {n} coefficients")

    pos = np.arange(n, dtype=np.float64)
    edges = np.linspace(0.0, float(n), n_bands + 1)
    masks = np.zeros((n_bands, n), dtype=np.float32)

    if not soft:
        for b in range(n_bands):
            lo, hi = edges[b], edges[b + 1]
            masks[b] = ((pos >= lo) & (pos < hi)).astype(np.float32)
        masks[-1, -1] = 1.0  # the top edge is exclusive; keep the last coeff
    else:
        width = float(n) / n_bands
        for b in range(n_bands):
            centre = 0.5 * (edges[b] + edges[b + 1])
            d = np.abs(pos - centre) / width
            masks[b] = np.clip(0.5 * (1.0 + np.cos(np.pi * np.clip(d, 0.0, 1.0))), 0, 1)
        total = masks.sum(axis=0, keepdims=True)
        masks /= np.where(total > 0, total, 1.0)

    return np.tile(masks, (1, channels)) if channels > 1 else masks


def patch_dct(image: np.ndarray, patch: int = DEFAULT_PATCH) -> np.ndarray:
    """Per-block 2D DCT-II. `(H, W, C)` uint8 or float -> `(nh, nw, C, p, p)`.

    Any partial block at the right or bottom edge is dropped rather than padded:
    padding invents an edge that the DCT would then report as high-frequency
    energy, which is exactly the band the frequency branch reads. Crop sizes are
    multiples of 8 in practice, so this costs nothing at the sizes used.
    """
    if image.ndim != 3:
        raise ValueError(f"expected (H, W, C), got shape {image.shape}")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    h, w, c = arr.shape
    nh, nw = h // patch, w // patch
    if nh < 1 or nw < 1:
        raise ValueError(f"image {h}x{w} is smaller than one {patch}x{patch} block")

    blocks = arr[: nh * patch, : nw * patch].reshape(nh, patch, nw, patch, c)
    blocks = blocks.transpose(0, 2, 4, 1, 3)  # (nh, nw, c, p, p)

    d = dct_basis(patch).astype(np.float32)
    return np.einsum("ij,hwcjk,lk->hwcil", d, blocks, d, optimize=True)


def cell_pool(coeffs: np.ndarray, grid: int = DEFAULT_GRID) -> np.ndarray:
    """Adaptive average pool `(nh, nw, C, p, p)` -> `(grid*grid, C, p*p)`.

    Averages coefficient *magnitudes*, not signed coefficients -- see the module
    docstring. Boundaries follow `torch.nn.AdaptiveAvgPool2d`: output cell i
    covers input rows `[floor(i*nh/grid), ceil((i+1)*nh/grid))`, so cells overlap
    by at most one block when `nh` is not a multiple of `grid` and every input
    block lands in at least one cell.
    """
    nh, nw, c, p, _ = coeffs.shape
    mag = np.abs(coeffs).reshape(nh, nw, c, p * p)

    rows = _pool_edges(nh, grid)
    cols = _pool_edges(nw, grid)
    out = np.empty((grid, grid, c, p * p), dtype=np.float32)
    for i, (r0, r1) in enumerate(rows):
        band = mag[r0:r1].mean(axis=0)
        for j, (c0, c1) in enumerate(cols):
            out[i, j] = band[c0:c1].mean(axis=0)
    return out.reshape(grid * grid, c, p * p)


def _pool_edges(n: int, grid: int) -> list[tuple[int, int]]:
    return [
        (int(np.floor(i * n / grid)), max(int(np.ceil((i + 1) * n / grid)), int(np.floor(i * n / grid)) + 1))
        for i in range(grid)
    ]


def normalize(x: np.ndarray) -> np.ndarray:
    """log1p, applied at render time.

    DCT magnitudes span several orders of magnitude -- DC is O(1) while the top
    radial band is O(1e-3) on natural content -- and fp16 on disk would flush the
    small end toward zero, which is the end the frequency branch is for. log1p
    compresses the range into something fp16 represents evenly and is monotone,
    so band orderings survive it. Applied after pooling so the average is of
    magnitudes rather than of logs.
    """
    return np.log1p(np.asarray(x, dtype=np.float32))


def extract_freq(
    image: np.ndarray,
    patch: int = DEFAULT_PATCH,
    grid: int = DEFAULT_GRID,
    radial: bool = True,
) -> np.ndarray:
    """The whole pipeline. `(H, W, C)` -> `(grid*grid, C * patch**2)` float32.

    Output shape is fixed by `(grid, patch, C)` alone and is independent of the
    input size, which is what lets one cached frequency view serve crops drawn
    anywhere in the 128-512px range. Channel-major layout: channel `c`'s
    coefficient `j` sits at `c * patch**2 + j`, and within a channel the
    coefficients are in radial order, so a band is a contiguous slice and
    `band_masks(..., channels=C)` lines up with it directly.
    """
    coeffs = patch_dct(image, patch)
    pooled = cell_pool(coeffs, grid)            # (cells, C, p*p)
    if radial:
        pooled = pooled[:, :, radial_order(patch)]
    cells, c, n = pooled.shape
    return normalize(pooled.reshape(cells, c * n))


def freq_fingerprint(patch: int, grid: int, channels: int, radial: bool, norm: str) -> str:
    """Identity of the extraction protocol, for `CacheSpec.freq_sha`.

    The coefficient set is a render-time commitment -- unlike the view count,
    which is resumable, changing any of these re-renders the whole frequency
    cache -- so it is asserted on load beside the other fingerprints rather than
    trusted to match.
    """
    import hashlib

    payload = repr(("freq", patch, grid, channels, radial, norm)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()
