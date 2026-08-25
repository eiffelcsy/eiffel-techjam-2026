"""The detector contract.

Everything the harness needs from a detector: turn a PIL image into a tensor,
turn a batch of tensors into per-image logits. Detectors are frozen -- they are
loaded, evaluated, and never trained here.
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image


class FrozenDetector(nn.Module, ABC):
    name: str = "detector"

    def __init__(self):
        super().__init__()

    def freeze(self) -> "FrozenDetector":
        """Eval mode, no gradients. Call at the end of a subclass __init__."""
        self.eval()
        self.requires_grad_(False)
        return self

    @abstractmethod
    def preprocess(self, img: Image.Image) -> torch.Tensor:
        """PIL image -> model input tensor (CHW). Keep the repo's own transform."""

    def preprocess_fn(self) -> Callable[[Image.Image], torch.Tensor]:
        """A picklable, model-free callable equivalent to `preprocess`.

        Datasets must never hold a reference to the model. `self.preprocess` is
        a bound method, so handing it to a Dataset drags every parameter along
        when DataLoader forks its workers -- which fails outright once the
        model is on MPS or CUDA ("_share_filename_: only available on CPU") and
        wastes memory even on CPU.

        The default is fine for detectors whose preprocessing needs nothing
        heavy. Subclasses carrying real weights override this to return a
        callable closing over only what preprocessing actually uses.
        """
        return self.preprocess

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch of inputs -> (B,) logits. Higher = more likely generated."""

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Logits -> probabilities in [0, 1].

        `forward` returns a fake-vs-real logit difference, so sigmoid of it is
        exactly the softmax mass on the generated classes.
        """
        if self.training:
            raise RuntimeError(
                f"{type(self).__name__} is in training mode; detectors are evaluated frozen. "
                "Did the subclass __init__ forget to call self.freeze()?"
            )
        return torch.sigmoid(self.forward(x))
