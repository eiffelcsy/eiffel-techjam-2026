"""The auxiliary-tensor pathway, and the two things it must not break.

`FusedDetector` is the only detector in this tree that reads the image twice, and
supporting it meant touching three of the harness's tensor-only sites -- the
dataset's return, `collate`, and the runner's `batch.to(device)`. Two claims come
out of that and both are tested here:

1. **The default path is unchanged.** A detector that returns no `aux_fn` must
   produce a bare stacked tensor exactly as before, with no `Inputs` anywhere
   near it. Every baseline in the project is scored through that path, and a
   change to it would silently move the denominator every GRACE number is read
   against.

2. **Both branches see one window.** On the training path that is free -- the
   dataset crops once and hands the result to both. On the evaluation arms the
   window is chosen inside preprocessing, from which no tensor can be read back,
   so the geometry is restated on the frequency side. Restated means it can
   disagree, so it is pinned here.

The real `FusedDetector` needs licence-gated DINOv3 weights, so what runs here is
a toy detector wired the same way. What these tests check is the plumbing --
which pixels, which tensors, which shapes -- and a real trunk is not needed to
get that wrong.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from grace.config import EnricherConfig, FreqConfig
from grace.models.factory import build_enricher
from grace.splits.base import FeatureSpec
from pipeline.data.dataset import AIGCDataset, Inputs, collate
from pipeline.degrade.crop import fixed_crop, fixed_resample
from pipeline.detectors.base import FrozenDetector
from pipeline.detectors.hf import _CropResizePreprocess, _ResamplePreprocess
from pipeline.freq.view import FreqExtract
from tests.fixtures import ToyPreprocess, write_images

SPEC = FeatureSpec(layout="vector", shape=(16,))
FREQ = FreqConfig(enabled=True, patch=4, grid=3)   # (9, 48)


class _PlainDetector(FrozenDetector):
    """One tensor in, one logit out. The harness's original shape."""

    name = "plain"

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(8, 1)
        self.freeze()

    def preprocess(self, img):
        return ToyPreprocess()(img)

    def preprocess_fn(self):
        return ToyPreprocess()

    def forward(self, x):
        return self.body(x.reshape(x.shape[0], -1)[:, :8]).squeeze(-1)


class _FusedToy(_PlainDetector):
    """The same detector plus a frequency branch, wired as `FusedDetector` is."""

    name = "fused-toy"

    def __init__(self, enricher=None):
        super().__init__()
        self.trunk_proj = nn.Linear(8, SPEC.dim)
        self.head = nn.Linear(SPEC.dim, 1)
        self.enricher = enricher if enricher is not None else _null_enricher()
        self._aux = FreqExtract(patch=FREQ.patch, grid=FREQ.grid)
        self.freeze()

    def aux_fn(self):
        return self._aux

    def features(self, x):
        return self.trunk_proj(x.reshape(x.shape[0], -1)[:, :8])

    def forward(self, x):
        if not hasattr(x, "aux"):
            raise TypeError("needs (image, frequency) inputs")
        f = self.features(x.x)
        return self.head(self.enricher(f, x.aux.float())).squeeze(-1)


def _null_enricher():
    return build_enricher(
        SPEC, FREQ.feature(), EnricherConfig(),
        patch=FREQ.patch, channels=FREQ.channels,
    )


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    return write_images(tmp_path_factory.mktemp("fused") / "images", 6)


# ------------------------------------------------- 1. the default path holds --


def test_a_single_tensor_detector_still_gets_a_bare_tensor(manifest):
    """Not an `Inputs` of one, not a tuple -- the same object `torch.stack`
    produced before any of this existed."""
    detector = _PlainDetector()
    dataset = AIGCDataset(
        manifest, preprocess=detector.preprocess_fn(), aux=detector.aux_fn()
    )
    batch, metas = collate([dataset[i] for i in range(4)])
    assert type(batch) is torch.Tensor
    assert batch.shape == (4, 3, 8, 8)
    assert len(metas) == 4
    detector.score(batch)          # the runner's one line, unchanged


def test_aux_fn_defaults_to_none():
    """The default on the base class is what keeps every other detector -- and
    every detector anyone adds later -- on the original path without opting in."""
    assert _PlainDetector().aux_fn() is None


def test_the_dataset_is_byte_identical_with_and_without_an_aux_of_none(manifest):
    """`aux=None` must not be a slightly different code path that happens to
    agree; it must be the same one."""
    preprocess = ToyPreprocess()
    plain = AIGCDataset(manifest, preprocess=preprocess)
    explicit = AIGCDataset(manifest, preprocess=preprocess, aux=None)
    for i in range(len(manifest)):
        assert torch.equal(plain[i][0], explicit[i][0])
        assert plain[i][1] == explicit[i][1]


# ------------------------------------------------------- 2. the aux pathway --


def test_the_aux_tensor_reaches_the_detector(manifest):
    detector = _FusedToy()
    dataset = AIGCDataset(
        manifest, preprocess=detector.preprocess_fn(), aux=detector.aux_fn()
    )
    batch, _ = collate([dataset[i] for i in range(4)])
    assert isinstance(batch, Inputs)
    assert batch.x.shape == (4, 3, 8, 8)
    assert batch.aux.shape == (4, *FREQ.shape)
    assert detector.score(batch).shape == (4,)


def test_inputs_moves_both_tensors(manifest):
    """`batch.to(device)` is the runner's whole interface to a batch."""
    batch = Inputs(torch.zeros(2, 3, 8, 8), torch.zeros(2, *FREQ.shape))
    moved = batch.to(torch.float64)
    assert isinstance(moved, Inputs)
    assert moved.x.dtype == torch.float64 and moved.aux.dtype == torch.float64


def test_a_fused_detector_refuses_a_bare_tensor(manifest):
    """Defaulting to zeros would score the enricher over nothing and report it
    under the name of the arm that reads the spectrum."""
    detector = _FusedToy()
    with pytest.raises(TypeError):
        detector.score(torch.zeros(2, 3, 8, 8))


def test_the_aux_extractor_is_picklable():
    """It is forked -- or under `spawn`, pickled -- into DataLoader workers, and
    it must carry no model. Same contract as `preprocess_fn`."""
    import pickle

    extractor = _FusedToy().aux_fn()
    assert len(pickle.dumps(extractor)) < 1024


# --------------------------------------------------- 3. the null enricher IS --


def test_the_null_enricher_reproduces_the_unenriched_arm(manifest):
    """E10 at the module level: a freshly built enricher must leave the logits
    bit-identical to the same detector without one.

    Exact equality, on real DCT tokens from real images -- not `allclose`, and
    not on zeros. This is the check that makes every later GRACE-freq number
    attributable to what the enricher learned.
    """
    detector = _FusedToy()
    dataset = AIGCDataset(
        manifest, preprocess=detector.preprocess_fn(), aux=detector.aux_fn()
    )
    batch, _ = collate([dataset[i] for i in range(len(manifest))])
    assert float(batch.aux.abs().sum()) > 0     # the tokens are not zeros

    with torch.no_grad():
        enriched = detector(batch)
        unenriched = detector.head(detector.features(batch.x)).squeeze(-1)
    assert torch.equal(enriched, unenriched)


# --------------------------------------- 4. one window, on both eval arms ----


@pytest.mark.parametrize(
    "geometry,size,reference",
    [("crop", 200, fixed_crop), ("resample", 512, fixed_resample)],
)
def test_the_frequency_branch_takes_the_arm_s_own_window(geometry, size, reference):
    """`FreqExtract.window` and the arm's preprocessing must select the same
    pixels. They are two restatements of one geometry -- the spatial branch reads
    it through `dinov3.VIEWS`, the frequency branch through `_freq_geometry` --
    and this is where they are pinned to each other."""
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (640, 480, 3), dtype=np.uint8))
    window = FreqExtract(geometry=geometry, size=size).window(img)
    assert window.size == reference(img, size).size
    assert np.array_equal(np.asarray(window), np.asarray(reference(img, size)))


def test_the_preprocessors_and_the_extractor_agree_on_the_window():
    """The stronger form of the check above: the same PIL image goes into the
    detector's real preprocessing and into the extractor, and the window the
    second takes is the one the first cropped."""
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (700, 900, 3), dtype=np.uint8))
    for geometry, size, cls in [
        ("crop", 200, _CropResizePreprocess), ("resample", 512, _ResamplePreprocess),
    ]:
        seen = {}

        class _Spy:
            def __call__(self, image, **kwargs):
                seen["window"] = image
                return type("_R", (), {"pixel_values": [torch.zeros(3, 4, 4)]})()

        cls(_Spy(), size)(img)
        window = FreqExtract(geometry=geometry, size=size).window(img)
        assert np.array_equal(np.asarray(seen["window"]), np.asarray(window)), geometry


def test_no_geometry_means_the_window_was_chosen_upstream():
    """The training path. The dataset has already drawn a `multiscale_crop`, so
    the extractor must read exactly what it is handed -- applying a second window
    here would crop a crop."""
    rng = np.random.default_rng(2)
    img = Image.fromarray(rng.integers(0, 256, (160, 200, 3), dtype=np.uint8))
    assert FreqExtract().window(img) is img


def test_the_output_shape_is_fixed_across_the_crop_range():
    """What lets one cached view serve every scale the draw produces, and what
    lets the tokens stack in `collate` with no special handling."""
    rng = np.random.default_rng(3)
    extract = FreqExtract(patch=FREQ.patch, grid=FREQ.grid)
    for side in (128, 200, 333, 512):
        img = Image.fromarray(rng.integers(0, 256, (side, side, 3), dtype=np.uint8))
        assert extract(img).shape == FREQ.shape, side


# ------------------------------------- 5. the geometry table has one owner ----


@pytest.mark.parametrize(
    "input_mode,expected",
    [
        ("crop200", ("crop", 200)),
        ("resample512", ("resample", 512)),
        # Not eval arms: the window is already chosen upstream (the dataset drew
        # it) or there is none, so the frequency branch reads what it is handed.
        ("multiscale", ("", 0)),
        ("resize", ("", 0)),
        ("crop", ("", 0)),
    ],
)
def test_freq_geometry_is_read_off_the_arm_s_own_table(input_mode, expected):
    """`FusedDetector` derives the DCT window from `dinov3.VIEWS`, the same table
    `_build_preprocess` reads. Two config keys that were meant to agree is how
    the two branches would end up on different windows."""
    from grace.detectors.fused import _freq_geometry

    detector = type("_Stub", (), {"input_mode": input_mode})()
    assert _freq_geometry(detector) == expected


def test_a_detector_with_no_input_mode_reads_what_it_is_handed():
    """Every detector but DINOv3's is gone from this tree, and the next one to
    arrive will not have an `input_mode`. Defaulting to "no geometry" is right:
    it means the caller chose the window, which is the training-path contract."""
    from grace.detectors.fused import _freq_geometry

    assert _freq_geometry(object()) == ("", 0)
