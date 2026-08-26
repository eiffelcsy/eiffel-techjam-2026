"""Generic adapter for any Hugging Face image-classification detector.

One class covers most published AIGC detectors on the Hub -- the model id goes
in config, nothing goes in code. The only thing that cannot be assumed is which
output index means *generated*: Hub detectors disagree, and a fair number order
`id2label` with the fake class at index 0. Guessing wrong yields a
plausible-looking (1 - AUC), so the index is resolved from the model's own
label map and the ambiguous cases raise instead of picking.
"""

import torch
from PIL import Image

from pipeline.detectors.base import FrozenDetector

DEFAULT_FAKE_LABELS = (
    "artificial", "fake", "ai", "ai_generated", "aigenerated",
    "generated", "synthetic",
)
# Deliberately excludes bare "0"/"1" and "LABEL_0"/"LABEL_1": a model whose head
# is unlabelled gives no evidence for which index is which, and raising is the
# honest outcome there.


def resolve_fake_indices(id2label: dict, fake_labels) -> list[int]:
    """Output indices whose label means *generated*.

    Raises rather than falling back to a positional default: an unrecognised
    label map is a configuration error the caller must fix with `fake_labels`,
    not something to paper over with a guess.
    """
    wanted = {str(w).lower().replace(" ", "_").replace("-", "_") for w in fake_labels}
    fake, real = [], []
    for idx, label in id2label.items():
        key = str(label).lower().replace(" ", "_").replace("-", "_")
        (fake if key in wanted else real).append(int(idx))

    shown = {int(k): v for k, v in id2label.items()}
    if not fake:
        raise ValueError(
            f"no output label matched fake_labels={sorted(wanted)}. Model labels are "
            f"{shown}. Pass the generated-class name(s) explicitly via args.fake_labels."
        )
    if not real:
        raise ValueError(
            f"every output label matched fake_labels={sorted(wanted)}. Model labels are "
            f"{shown}. There is no real class left to score against."
        )
    return sorted(fake)


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
        w, h = img.size
        if w < self.size or h < self.size:
            scale = self.size / min(w, h)
            img = img.resize(
                (max(self.size, round(w * scale)), max(self.size, round(h * scale))),
                Image.BICUBIC,
            )
            w, h = img.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
        return self.processor(img, do_resize=False, return_tensors="pt").pixel_values[0]


class HFImageClassifier(FrozenDetector):
    """Any `AutoModelForImageClassification` checkpoint, used as a frozen detector.

    Parameters
    ----------
    model_id    : hub id or local path
    fake_labels : label names meaning *generated*; override for idiosyncratic heads
    name        : display name in results; defaults to the model id's last segment
    """

    def __init__(
        self,
        model_id: str,
        fake_labels=DEFAULT_FAKE_LABELS,
        revision: str | None = None,
        name: str | None = None,
    ):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.model_id = model_id
        self.name = name or model_id.rstrip("/").split("/")[-1]
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, revision=revision)

        id2label = self.model.config.id2label
        self.fake_idx = resolve_fake_indices(id2label, fake_labels)
        self.real_idx = sorted(set(int(i) for i in id2label) - set(self.fake_idx))
        self._preprocess = _ProcessorPreprocess(self.processor)
        self.freeze()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._preprocess(img)

    def preprocess_fn(self):
        return self._preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        fake = torch.logsumexp(logits[:, self.fake_idx], dim=1)
        real = torch.logsumexp(logits[:, self.real_idx], dim=1)
        return fake - real
