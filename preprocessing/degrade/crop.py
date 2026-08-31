"""Multi-scale cropping: the view selection that sits between degradation and
preprocessing.

    load_normalized  ->  condition (degrade at native res)  ->  CROP  ->  preprocess

Why the crop exists. WildFake ships its COCO reals pre-resized to a uniform
200x200 while its DALL-E 3 fakes are native 1024px output, so on the reported
benchmark `max(w, h) > 512` separates the classes at TPR 0.9994 / FPR 0.0000 --
image dimensions alone score ~0.9997 AUC. Any model shown whole images reads
that channel, and a frequency branch reads it hardest. Cropping at native pixel
scale removes it: the model never sees a whole image, so there is no global
resolution to read, and the traces it must find are scale-local.

The trade is deliberate and it costs real signal. Global composition, whole-image
colour statistics and aspect ratio are all gone. Generation traces are a local,
high-frequency phenomenon at native pixel scale, which is exactly what a
128-512px native crop preserves and what a whole-image resize to 224 destroys, so
the trade runs in the direction this project cares about -- but a crop-trained
detector is not comparable like-for-like against a whole-image one, and a
degradation whose damage is itself global is only seen through its local residue.

Why the crop is not a transform. `ops.TRANSFORMS` are size-preserving by
contract, which is what lets them compose in any order and lets one grid apply to
a dataset of mixed resolutions. A crop is a view selection, not a degradation, so
it lives here and is applied after the recipe rather than inside it. Degradation
therefore still happens at native resolution, unchanged: `ops.py`'s "parameters
are applied identically regardless of image size" still holds, and the existing
parameter grid keeps its calibration.

Why the draw is seeded rather than random. `cache.spec.sha_preprocess` runs a
preprocessing transform twice and refuses a stochastic one, because the feature
cache is only meaningful if a row is reproducible. The degradation schedule
already solved this -- `stable_seed(index, level, replicate, seed)` makes each
epoch's corruption reproducibly random -- and the crop follows the same pattern
with its own `"crop"` tag, so the crop draw is independent of the recipe draw and
of the noise draw while staying a pure function of its inputs.

Randomness is training-only. Both evaluation arms are deterministic:

    arm (a)   fixed_crop(200)       a 200x200 window at native pixel scale
    arm (b)   fixed_resample(512)   the whole image resampled to 512x512

Within either arm every image has identical dimensions, so the dimension
shortcut is 0.5 by construction rather than by normalisation. A per-condition
crop draw would break the pairing the retention denominator depends on -- it
compares the same image clean vs degraded across all 26 conditions -- which is
why eval draws nothing.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from common.seeding import stable_seed

POLICIES = ("uniform", "log_uniform")
"""How the crop side length is drawn from [s_min, s_max].

    uniform      equal weight per pixel of side length
    log_uniform  equal weight per octave of scale

`uniform` is the default and puts most of its mass at the coarse end of the
range: drawing s in [128, 512] uniformly means three quarters of crops are
larger than 224 and are therefore downsampled by preprocessing, which attenuates
the high-frequency traces the crop existed to preserve. `log_uniform` spreads
the mass evenly across scale instead. Kept as a one-key ablation (E16) rather
than a default, since the plan's protocol table specifies uniform.
"""


@dataclass(frozen=True)
class CropDraw:
    """What was actually taken, recorded per row so it can be audited later.

    `clamped` is the important field. If the drawn side exceeds the source's
    short side there is no window that large to take, and the size is reduced to
    fit rather than the source being upscaled to meet it. Upscaling would hide
    the problem: a class whose sources are small would be silently upsampled and
    a frequency branch would read the interpolation instead of the label.
    Clamping instead makes the shortfall visible in `recipes.parquet`, which is
    what lets E-cropsize measure whether crop size has become class-conditional
    -- i.e. whether crop size has itself become the label.
    """

    size: int       # the side actually taken
    left: int
    top: int
    drawn: int      # the side the policy asked for, before clamping
    clamped: bool

    @property
    def scale(self) -> float:
        """Side length relative to what was asked for. 1.0 unless clamped."""
        return self.size / self.drawn


def draw_size(rng: np.random.Generator, s_min: int, s_max: int, policy: str) -> int:
    """The size draw alone, exposed so the range audit can mirror it exactly.

    `scripts/misc/audit_sizes.py` decides the training range by simulating this over
    a corpus's header-read sizes, without decoding a pixel. It has to be the
    same draw the renderer performs or the audit certifies a protocol nobody
    runs, so it is one function rather than two that agree by inspection.
    """
    if policy == "uniform":
        return int(rng.integers(s_min, s_max + 1))
    if policy == "log_uniform":
        return int(round(float(np.exp(rng.uniform(np.log(s_min), np.log(s_max + 1))))))
    raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")


def multiscale_crop(
    img: Image.Image,
    index: int,
    epoch: int,
    seed: int = 0,
    s_min: int = 128,
    s_max: int = 512,
    policy: str = "uniform",
) -> tuple[Image.Image, CropDraw]:
    """Draw one square window, deterministically in `(index, epoch, seed)`.

    `epoch` plays the role the degradation schedule gives it: the same image
    gets a different crop each epoch, which is the augmentation, but the same
    `(index, epoch)` gives a bit-identical crop in every process and every run,
    which is what makes the rendered feature cache reproducible.

    Draw order -- size, then left, then top -- is part of the contract. Changing
    it changes every crop in the corpus without changing any fingerprint, so it
    must not be reordered casually; `crop_sha` covers the parameters, not the
    order.
    """
    if s_min < 1 or s_max < s_min:
        raise ValueError(f"need 1 <= s_min <= s_max, got s_min={s_min}, s_max={s_max}")

    rng = np.random.default_rng(stable_seed(index, epoch, seed, "crop"))
    drawn = draw_size(rng, s_min, s_max, policy)

    w, h = img.size
    size = min(drawn, w, h)
    left = int(rng.integers(0, w - size + 1))
    top = int(rng.integers(0, h - size + 1))

    window = img.crop((left, top, left + size, top + size))
    return window, CropDraw(size, left, top, drawn, size < drawn)


SAMPLE_EPOCH = 0
"""The epoch every training-time crop is drawn at, i.e. the window is fixed per
image and does not vary across epochs.

This is forced by what the cache pairs. Stage 1 trains `adapter(f_deg) -> f_clean`
on the same row, so the degraded and clean features must be of the *same window*
-- otherwise the adapter is asked to map one window onto a different one, which
is not a restoration task. The cache stores one clean view per image and N
degraded views, so a per-epoch window would need a clean view per epoch too.

Stage 0 settles it independently: `probe.train.extract_features` runs the trunk
once and fits the head over the cached features for 40 epochs, so a per-epoch
crop there would cost 40 trunk passes. The architecture already commits to one
window per image at that stage.

So the epoch varies the *degradation* and the image index varies the *window*.
Multi-scale diversity comes from across the corpus -- 100k images at 100k
independently drawn scales -- rather than from re-drawing per epoch. If stage-1
generalisation ever looks window-limited, the alternative is per-epoch crops
with a matching clean view per epoch, which costs ~1.8 GB of extra spatial cache
and a change to the one-clean-view layout.
"""


class SampleCrop:
    """Picklable `(image, index) -> image` for the render and training paths.

    A module-level class rather than a closure for the same reason
    `detectors.hf._CropPreprocess` is one: it is handed to a Dataset that gets
    forked into DataLoader workers, and it must carry no model and no open
    handles. Holds four ints and a string.
    """

    def __init__(
        self, s_min: int = 128, s_max: int = 512, seed: int = 0, policy: str = "uniform"
    ):
        self.s_min, self.s_max = int(s_min), int(s_max)
        self.seed, self.policy = int(seed), str(policy)

    def __call__(self, img: Image.Image, index: int) -> Image.Image:
        window, _ = multiscale_crop(
            img, index, SAMPLE_EPOCH, self.seed, self.s_min, self.s_max, self.policy
        )
        return window

    def draw(self, img: Image.Image, index: int) -> CropDraw:
        """The draw without the pixels, for recording into `recipes.parquet`."""
        return multiscale_crop(
            img, index, SAMPLE_EPOCH, self.seed, self.s_min, self.s_max, self.policy
        )[1]

    def fingerprint(self) -> str:
        return crop_fingerprint(self.s_min, self.s_max, self.seed, self.policy)


def fixed_crop(img: Image.Image, size: int = 200) -> Image.Image:
    """Centre window of `size`, at native pixel scale. Evaluation arm (a).

    Deterministic and index-free, so it composes into a detector's `preprocess`
    and passes `sha_preprocess`. This is `detectors.hf._CropPreprocess`'s
    geometry without its normalisation step: a source shorter than `size` on
    either axis is scaled up preserving aspect until it fits, then cropped. On
    the reported benchmark at size=200 that upscale never fires -- the reals are
    exactly 200x200 and the fakes are all larger -- which is precisely why 200 is
    the arm's size. It is the largest window every image in the corpus can supply
    from its own pixels.
    """
    w, h = img.size
    if w < size or h < size:
        scale = size / min(w, h)
        img = img.resize(
            (max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC
        )
        w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def fixed_resample(img: Image.Image, size: int = 512) -> Image.Image:
    """The whole image squashed to `size` x `size`. Evaluation arm (b).

    Square-squashed rather than aspect-preserving on purpose: this is what the
    HF processor's own transform does (`default_to_square: true`), so arm (b) is
    the existing `input_mode: resize` protocol made explicit and doubles as a
    continuity arm against the baseline configs. It distorts the 12.4% of fakes
    that are 1792x1024.

    Read arm (b) as a scale-robustness check on the spatial branch, not as
    evidence about frequency. With WildFake's 200x200 reals it upsamples reals
    2.56x while downsampling fakes 2x, so the reals have near-zero energy above
    200-Nyquist by construction and the spectral-rolloff floor (E-rolloff) sits
    very high here. Arm (a) is the informative one for the frequency branch.
    """
    return img.resize((size, size), Image.BICUBIC)


def crop_fingerprint(
    s_min: int, s_max: int, seed: int, policy: str, enabled: bool = True
) -> str:
    """Identity of the crop protocol, for `CacheSpec.crop_sha`.

    A cache rendered under one crop protocol is not interchangeable with one
    rendered under another -- the features are of different windows of different
    images -- so this is asserted on load beside the manifest, schedule, detector
    and preprocess hashes. `enabled=False` fingerprints "no crop at all", which
    is what a whole-image cache carries, so the two can never be silently mixed.
    """
    import hashlib

    payload = repr(("crop", enabled, s_min, s_max, seed, policy)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()
