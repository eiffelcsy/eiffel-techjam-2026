"""The six transforms, each with a fixed parameter grid.

Every transform is a real-world image-handling artefact, not an adversarial
attack -- the question this harness asks is whether a detector survives an
image having been posted, resized, filtered, or screenshotted.

    transform      param            values                  real-world analog
    ------------------------------------------------------------------------
    jpeg           quality          90, 70, 50, 30          social re-encode
    gaussian_blur  sigma            0.5, 1.0, 2.0           out-of-focus
    resize         scale            0.5, 0.25               thumbnailing
    gaussian_noise sigma            0.02, 0.05, 0.10        low-light sensor
    color_jitter   strength         0.2                     filter / auto-enhance
    center_crop    keep             0.8                     profile-pic framing

Rules: a transform is pure (randomness only from the rng it is handed) and
returns an image the same mode *and size* as it got, at any input resolution.
Size preservation is what lets steps compose in any order and lets one grid
apply unchanged to a dataset of mixed resolutions. gaussian_noise and
color_jitter draw from the rng; the other four are deterministic given their
parameter.

Parameters are applied identically regardless of image size -- no clamping, no
minimum-size guard. Some are resolution-relative (`scale`, `keep`) and some
absolute (`sigma` in pixels), which is a property of the artefacts themselves.
"""

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

TransformFn = Callable[[Image.Image, Any, np.random.Generator], Image.Image]


@dataclass(frozen=True)
class Transform:
    name: str
    group: str          # compression | blur | resampling | noise | photometric | framing
    param_name: str     # "quality", "sigma", "scale", "strength", "keep"
    params: tuple       # the grid, ordered mild -> severe
    analog: str         # one-line real-world justification, printed in the report
    fn: TransformFn


TRANSFORMS: dict[str, Transform] = {}


def register(name: str, group: str, param_name: str, params: tuple, analog: str):
    """Decorator adding a transform fn to TRANSFORMS."""

    def decorate(fn: TransformFn) -> TransformFn:
        TRANSFORMS[name] = Transform(name, group, param_name, tuple(params), analog, fn)
        return fn

    return decorate


def _at_least_1px(*dims: float) -> tuple[int, ...]:
    """Degenerate sizes are clamped to 1px, not skipped: the recipe still applies."""
    return tuple(max(1, int(round(d))) for d in dims)


# --- the six -----------------------------------------------------------------

@register("jpeg", "compression", "quality", (90, 70, 50, 30), "social re-encode, messaging")
def jpeg(img, quality: int, rng):
    """Re-encode as JPEG at the given quality and decode back."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        out.load()
        return out.convert(img.mode)


@register("gaussian_blur", "blur", "sigma", (0.5, 1.0, 2.0), "out-of-focus capture")
def gaussian_blur(img, sigma: float, rng):
    """Gaussian blur, kernel radius derived from sigma."""
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


@register("resize", "resampling", "scale", (0.5, 0.25), "thumbnail generation")
def resize(img, scale: float, rng):
    """Downscale by `scale`, then back up to the original size (bilinear)."""
    w, h = img.size
    small = img.resize(_at_least_1px(w * scale, h * scale), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


@register("gaussian_noise", "noise", "sigma", (0.02, 0.05, 0.10), "low-light sensor noise")
def gaussian_noise(img, sigma: float, rng):
    """Additive Gaussian noise, sigma in [0,1] units of pixel range."""
    arr = np.asarray(img, dtype=np.float32)
    noisy = arr + rng.normal(0.0, float(sigma) * 255.0, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode=img.mode)


@register("color_jitter", "photometric", "strength", (0.2,), "filter apps, auto-enhance")
def color_jitter(img, strength: float, rng):
    """Brightness / contrast / saturation each scaled by U(1-s, 1+s)."""
    s = float(strength)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(float(rng.uniform(1.0 - s, 1.0 + s)))
    return img


@register("center_crop", "framing", "keep", (0.8,), "profile-picture cropping, framing")
def center_crop(img, keep: float, rng):
    """Keep the central `keep` fraction of each side, then resize back up."""
    w, h = img.size
    cw, ch = _at_least_1px(w * keep, h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)
