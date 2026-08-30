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

    def aux_fn(self) -> Callable[[Image.Image], torch.Tensor] | None:
        """A SECOND read of the same image, or None -- which is the default and
        the case for every detector but one.

        Returning None keeps the harness on its original path exactly: the
        dataset yields a bare tensor, `collate` stacks it, and `forward` receives
        what it always received. Returning a callable makes `forward` receive a
        `preprocessing.dataset.Inputs` instead.

        It exists because some information cannot be recovered downstream of
        preprocessing at any cost. `freq_branch.detectors.fused.FusedDetector` needs a
        patch-DCT at native pixel scale, and the 224px normalized tensor no
        longer contains it. Same picklability contract as `preprocess_fn`: the
        callable is forked into DataLoader workers and must hold no model.
        """
        return None

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
