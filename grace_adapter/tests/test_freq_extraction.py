"""The frequency extractor must be one implementation, exactly reproducible.

The render path and the evaluation path both call `extract_freq`; if they can
disagree, the cached frequency view is a different feature from the one the
detector sees and nothing downstream would catch it. The E0 falsification gate
calls the same function, so a band separation measured there is a statement
about the shipped model rather than about a script.

The load-bearing property is the fixed output shape: crops vary over 128-512px
and every one of them must produce `(196, 192)`, or the cache has no layout.
"""

import numpy as np
import pytest

from pipeline.freq.dct import (
    DEFAULT_GRID, DEFAULT_PATCH,
    band_masks, cell_pool, dct_basis, extract_freq, freq_fingerprint,
    normalize, patch_dct, radial_order,
)

N_COEFFS = 3 * DEFAULT_PATCH * DEFAULT_PATCH   # 192
N_CELLS = DEFAULT_GRID * DEFAULT_GRID          # 196


def image(size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


# --- the basis is really a DCT ----------------------------------------------

def test_basis_is_orthonormal():
    d = dct_basis(8)
    assert np.allclose(d @ d.T, np.eye(8), atol=1e-12)


def test_basis_matches_scipy():
    """Written out by hand so the extractor needs only numpy in DataLoader
    workers; scipy is the oracle, not the dependency."""
    fft = pytest.importorskip("scipy.fft")
    block = np.random.default_rng(0).standard_normal((8, 8))
    d = dct_basis(8)
    assert np.allclose(d @ block @ d.T, fft.dctn(block, norm="ortho"), atol=1e-12)


def test_dc_coefficient_is_the_block_mean():
    flat = np.full((8, 8, 3), 128, dtype=np.uint8)
    coeffs = patch_dct(flat)
    assert coeffs.shape == (1, 1, 3, 8, 8)
    assert coeffs[0, 0, 0, 0, 0] == pytest.approx(8 * (128 / 255.0), rel=1e-5)
    assert np.allclose(coeffs[0, 0, :, 1:, :], 0, atol=1e-6)
    assert np.allclose(coeffs[0, 0, :, :, 1:], 0, atol=1e-6)


# --- shapes and determinism --------------------------------------------------

def test_patch_dct_shape():
    assert patch_dct(image(128)).shape == (16, 16, 3, 8, 8)
    assert patch_dct(image(512)).shape == (64, 64, 3, 8, 8)


def test_partial_blocks_are_dropped_not_padded():
    """Padding invents an edge, and the DCT reports an invented edge as
    high-frequency energy -- the exact band the frequency branch reads."""
    assert patch_dct(image(130)).shape == (16, 16, 3, 8, 8)


def test_an_image_smaller_than_one_block_is_an_error():
    with pytest.raises(ValueError, match="smaller than one"):
        patch_dct(image(4))


def test_a_flat_array_is_rejected():
    with pytest.raises(ValueError, match=r"\(H, W, C\)"):
        patch_dct(np.zeros((64, 64), dtype=np.uint8))


def test_extraction_is_deterministic():
    img = image(256)
    assert np.array_equal(extract_freq(img), extract_freq(img))


@pytest.mark.parametrize("size", [128, 168, 224, 256, 384, 512])
def test_output_shape_is_fixed_across_the_whole_crop_range(size):
    """One cached layout has to serve every crop the multi-scale draw produces."""
    assert extract_freq(image(size)).shape == (N_CELLS, N_COEFFS)


def test_output_is_float32_and_finite():
    out = extract_freq(image(256))
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


# --- radial ordering ---------------------------------------------------------

def test_radial_order_is_a_permutation():
    order = radial_order(8)
    assert sorted(order.tolist()) == list(range(64))


def test_radial_order_starts_at_dc():
    assert radial_order(8)[0] == 0


def test_radial_order_is_monotone_in_frequency():
    """What makes a band a contiguous slice rather than a scatter of indices."""
    order = radial_order(8)
    u, v = np.divmod(order, 8)
    radius = np.sqrt(u**2 + v**2)
    assert np.all(np.diff(radius) >= -1e-12)


# --- band masks --------------------------------------------------------------

def test_bands_partition_the_spectrum():
    """Sum-to-one means the split routes evidence between experts rather than
    discarding any of it."""
    for n_bands in (2, 3, 4):
        masks = band_masks(8, n_bands)
        assert masks.shape == (n_bands, 64)
        assert np.allclose(masks.sum(axis=0), 1.0)


def test_hard_bands_are_disjoint():
    masks = band_masks(8, 2)
    assert np.all(masks[0] * masks[1] == 0)


def test_soft_bands_overlap_but_still_sum_to_one():
    masks = band_masks(8, 2, soft=True)
    assert np.allclose(masks.sum(axis=0), 1.0)
    assert np.any(masks[0] * masks[1] > 0)


def test_low_band_is_low_frequency():
    """Band 0 must be the DC end, or the HF and LF experts are swapped and every
    interpretation of their gates is backwards."""
    masks = band_masks(8, 2)
    assert masks[0][0] == 1.0
    assert masks[-1][-1] == 1.0


def test_masks_tile_across_channels():
    """A band has to mean the same frequencies in every colour channel, matching
    `extract_freq`'s channel-major layout."""
    one = band_masks(8, 2, channels=1)
    three = band_masks(8, 2, channels=3)
    assert three.shape == (2, 192)
    assert np.array_equal(three, np.tile(one, (1, 3)))
    assert np.allclose(three.sum(axis=0), 1.0)


def test_band_count_is_validated():
    with pytest.raises(ValueError, match="n_bands"):
        band_masks(8, 0)
    with pytest.raises(ValueError, match="exceeds"):
        band_masks(8, 65)


# --- pooling magnitudes, not signed coefficients -----------------------------

def test_cell_pool_averages_magnitudes():
    """Signed DCT coefficients have essentially arbitrary sign across blocks of
    natural content, so averaging them cancels toward zero and the feature dies.
    This is the single subtlest way the extractor could silently produce
    nothing."""
    coeffs = np.zeros((14, 14, 1, 8, 8), dtype=np.float32)
    coeffs[..., 3, 3] = np.where(
        (np.arange(14 * 14).reshape(14, 14, 1) % 2) == 0, 2.0, -2.0
    )
    pooled = cell_pool(coeffs, grid=1)
    assert pooled[0, 0, 3 * 8 + 3] == pytest.approx(2.0)


def test_cell_pool_shape_and_grid():
    coeffs = patch_dct(image(512))
    assert cell_pool(coeffs, grid=14).shape == (196, 3, 64)
    assert cell_pool(coeffs, grid=7).shape == (49, 3, 64)


def test_every_block_lands_in_a_cell():
    """Adaptive pooling with `nh` not a multiple of `grid` must not silently drop
    the remainder: 16 blocks into 14 cells is the 128px crop, the commonest case
    at the fine end of the range."""
    coeffs = np.ones((16, 16, 1, 8, 8), dtype=np.float32)
    pooled = cell_pool(coeffs, grid=14)
    assert pooled.shape == (196, 1, 64)
    assert np.allclose(pooled, 1.0)


def test_normalize_is_log1p_and_monotone():
    x = np.array([0.0, 0.5, 2.0, 100.0], dtype=np.float32)
    assert np.allclose(normalize(x), np.log1p(x))
    assert np.all(np.diff(normalize(x)) > 0)


# --- the mechanism the branch is built on ------------------------------------

def test_blur_removes_high_band_energy():
    """E0's premise in miniature: if a degradation that destroys high
    frequencies does not show up in the high band, the branch has no signal to
    read and the extractor is wrong."""
    from PIL import Image, ImageFilter

    sharp = Image.fromarray(image(256, seed=3), mode="RGB")
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=2.0))

    hf = band_masks(8, 2, channels=3)[1]
    a = (extract_freq(np.asarray(sharp)) * hf).sum()
    b = (extract_freq(np.asarray(blurred)) * hf).sum()
    assert b < a


def test_noise_adds_high_band_energy():
    """The opposite sign, which is why the enricher ships two experts with
    independent gates rather than one."""
    rng = np.random.default_rng(4)
    base = image(256, seed=4).astype(np.float32)
    noisy = np.clip(base + rng.normal(0, 25, base.shape), 0, 255).astype(np.uint8)

    hf = band_masks(8, 2, channels=3)[1]
    a = (extract_freq(base.astype(np.uint8)) * hf).sum()
    b = (extract_freq(noisy) * hf).sum()
    assert b > a


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_is_stable_and_parameter_sensitive():
    base = dict(patch=8, grid=14, channels=3, radial=True, norm="log1p")
    assert freq_fingerprint(**base) == freq_fingerprint(**base)
    assert freq_fingerprint(**{**base, "patch": 16}) != freq_fingerprint(**base)
    assert freq_fingerprint(**{**base, "grid": 28}) != freq_fingerprint(**base)
    assert freq_fingerprint(**{**base, "norm": "none"}) != freq_fingerprint(**base)
