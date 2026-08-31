"""Hub preprocessing, shared by anything built on an `AutoImageProcessor`.

`_ProcessorPreprocess` and `_CropPreprocess` are the two ways a source image
becomes a model input, and `dinov3.DINOv3MLPDetector` selects between them by
its `input_mode`. They live here rather than in that module because they are
properties of the Hub processor contract, not of DINOv3.
"""

import torch
from PIL import Image

from preprocessing.degrade.crop import fixed_crop, fixed_resample

class _ProcessorPreprocess:
    """Callable holding the image processor and nothing else.

    Deliberately a module-level class rather than a closure or a bound method:
    it must pickle cleanly into DataLoader workers, and it must not reach the
    model. See `FrozenDetector.preprocess_fn`.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.processor(img, return_tensors="pt").pixel_values[0]


class _CropPreprocess:
    """Crop at native resolution, then normalize. No resize.

    The processor's own transform resizes the whole image to the model's square
    input, which for a 1024px source is a ~4.6x downsample. That destroys the
    high-frequency generation traces a forensic detector exists to read: by the
    time the trunk sees the image, the only thing left to separate real from
    generated is *content*, and a probe on a semantic backbone will happily fit
    that instead. Cropping keeps pixels at the scale they were generated at.

    Center crop, not random: `sha_preprocess` runs the transform twice and
    refuses a stochastic one, because a cache keyed on a transform that returns
    different tensors for the same image is not a cache. The harness needs the
    same property for a different reason -- per-condition score tables line up
    row for row only if an image yields the same crop under every condition.

    Images smaller than the crop are scaled up to it. Padding would be the
    alternative and is worse: a border whose width is a function of the source
    resolution is exactly the container statistic the decode path exists to
    strip.
    """

    def __init__(self, processor, size: int):
        self.processor = processor
        self.size = int(size)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.processor(
            fixed_crop(img, self.size), do_resize=False, return_tensors="pt"
        ).pixel_values[0]


class _CropResizePreprocess:
    """Crop a window at native scale, then resize it to the model input.

    Evaluation arm (a). Differs from `_CropPreprocess` in one way that matters:
    the window is not the model's input size. A 200px window cannot be fed to a
    patch-16 ViT directly (200/16 is not an integer) and, more to the point, the
    window size is a property of the *dataset* -- 200 is the largest square every
    image in WildFake supplies from its own pixels, because every real in it is
    exactly 200x200 -- while 224 is a property of the model. Cropping at the
    former and resizing to the latter keeps the two decisions separate.

    Every image in this arm therefore has identical dimensions going in, which is
    what makes the dimension shortcut 0.5 by construction rather than by
    normalisation. On the reported benchmark the reals are passed through
    untouched and nothing is upsampled.
    """

    def __init__(self, processor, crop_size: int):
        self.processor = processor
        self.crop_size = int(crop_size)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.processor(
            fixed_crop(img, self.crop_size), return_tensors="pt"
        ).pixel_values[0]


class _ResamplePreprocess:
    """Squash the whole image to a fixed square, then to the model input.

    Evaluation arm (b), and materially the `resize` protocol made explicit: it
    is a continuity arm against the existing baselines rather than a new view.

    Read it as a scale-robustness check, not as evidence about frequency. It
    upsamples WildFake's 200x200 reals 2.56x while downsampling the fakes 2x, so
    the reals carry near-zero energy above 200-Nyquist by construction and the
    spectral-rolloff floor sits very high here. It is also out of distribution
    for a multi-scale-trained model, which never sees a whole image -- that is
    the point of the arm, but it means a drop here is a generalisation result and
    not a like-for-like comparison against arm (a).
    """

    def __init__(self, processor, size: int):
        self.processor = processor
        self.size = int(size)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.processor(
            fixed_resample(img, self.size), return_tensors="pt"
        ).pixel_values[0]
