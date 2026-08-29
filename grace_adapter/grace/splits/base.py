"""The trunk/head split: the one thing a detector must expose for GRACE.

`FrozenDetector` (in the eval harness) is deliberately opaque -- image in, logit
out -- because measurement needs nothing else. GRACE needs the seam: a feature
tensor it can cache, correct, and feed back. `SplitDetector` is that seam, added
*around* a detector rather than inside it, so the harness stays model-agnostic
and the zoo adapters stay untouched.

The contract is one equation, and `tests/test_split_consistency.py` enforces it:

    head(trunk(x)) == detector(x)      for every x, to float tolerance

Break it and every number downstream is measuring a different model than the
Day-1 baseline it is compared against.

`head` must be differentiable with respect to its *input*. GRACE takes the
gradient of the logit at the clean features to find the head's sensitive
subspace (see `grace.train.weighting`), so a head wrapped in `no_grad` or
`torch.inference_mode` silently disables the decision-weighted objective. The
parameters stay frozen; only the input needs a graph.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from pipeline.detectors.base import FrozenDetector

LAYOUTS = ("vector", "tokens", "layers")
"""What the trunk emits, per image, ignoring the batch dimension.

    vector  (D,)      one embedding                      -- DINOv3 (the only
                                                            seam in this tree)
    tokens  (T, D)    per-patch or per-window tokens      -- historical: B-Free
    layers  (L, D)    one CLS token per encoder block     -- historical: RINE

Only `vector` has a live detector behind it now. The other two are kept because
the adapter, the cache and the factory all branch on layout and are tested
across all three -- removing them would delete working generality, not dead
code, and would have to be rebuilt to adapt any seam that is not pooled.

`tokens` and `layers` are the same tensor rank and differ only in what the
adapter does with the group axis: tokens share one gate, layers get one gate
each. That is a `gate_shape` argument, not a class -- see
`grace.models.adapter`.
"""

_NDIM = {"vector": 1, "tokens": 2, "layers": 2}


@dataclass(frozen=True)
class FeatureSpec:
    """Shape and dtype of one image's trunk output.

    `dtype` is the *cache* dtype, not the compute dtype: features are stored
    float16 and cast to float32 before any loss touches them (fp16 MSE on
    unnormalized ViT features underflows to zero).
    """

    layout: str
    shape: tuple[int, ...]
    dtype: str = "float16"

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        if len(self.shape) != _NDIM[self.layout]:
            raise ValueError(
                f"layout {self.layout!r} expects a {_NDIM[self.layout]}-d shape, "
                f"got {self.shape}"
            )
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))

    @property
    def dim(self) -> int:
        """Channel width -- the axis the adapter's MLP operates on."""
        return self.shape[-1]

    @property
    def n_groups(self) -> int:
        """Size of the group axis: layers for `layers`, tokens for `tokens`, 1 for
        `vector`. The per-group gate vector and the per-group damage profile are
        both indexed by this."""
        return self.shape[0] if len(self.shape) == 2 else 1

    @property
    def ndim(self) -> int:
        """Rank of a *batched* feature tensor."""
        return len(self.shape) + 1

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def bytes_per_image(self) -> int:
        """For the size estimate in `scripts/build_cache.py --dry-run`."""
        return self.numel() * int(torch.empty(0, dtype=self.torch_dtype).element_size())

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def to_dict(self) -> dict:
        return {"layout": self.layout, "shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(layout=d["layout"], shape=tuple(d["shape"]), dtype=d.get("dtype", "float16"))


class SplitDetector(nn.Module, ABC):
    """A frozen detector, cut in two.

    Holds the detector rather than subclassing it: a split is a *view* of an
    already-built model, and `build_detector` stays the only way a detector is
    constructed. Never trains -- the wrapped model arrives frozen, and the
    training loop asserts that every step.
    """

    def __init__(self, detector: FrozenDetector):
        super().__init__()
        self.detector = detector

    @property
    def name(self) -> str:
        return getattr(self.detector, "name", "detector")

    @property
    @abstractmethod
    def feature_spec(self) -> FeatureSpec:
        """Declared, not inferred: the cache writer commits to it before the
        first batch. `tests/test_split_consistency.py` checks it against what
        `trunk` actually emits."""

    @abstractmethod
    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocessed batch -> (B, *feature_spec.shape). Frozen, no grad needed."""

    @abstractmethod
    def head(self, f: torch.Tensor) -> torch.Tensor:
        """(B, *feature_spec.shape) -> (B,) logits.

        Frozen, but gradient must flow *through* it to its input -- see the
        module docstring.
        """

    def taps(self) -> tuple[str, ...]:
        """Names of the intermediate activations this split exposes, in order.

        Empty means "no ladder": every consumer treats taps as absent and the
        split behaves exactly as it did before taps existed. A split that
        returns names must also override `tap_spec` and `trunk_with_taps`, and
        `verify_taps` checks that the three agree.

        The names are stored in `CacheSpec.taps` and are what a stale cache is
        diagnosed against, so they must identify *which* activation was read --
        `"block04"`, not `"tap0"`.
        """
        return ()

    def tap_spec(self) -> FeatureSpec | None:
        """Shape of one image's stacked taps, or None when there are none.

        Layout is `layers` and the shape is `(len(taps()), tap_dim)`: the taps
        are one tensor rather than a dict, because a dict of ragged widths would
        need its own cache view per tap and its own branch in the adapter, and
        every split worth tapping reads the same width at every depth anyway.

        Deliberately a `FeatureSpec`, the same type the seam uses: the cache
        writer, the shard reader and the size estimate then work on taps with no
        second code path.
        """
        return None

    def trunk_with_taps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Preprocessed batch -> (seam features, taps) in ONE forward pass.

        The taps are activations the trunk already computed and threw away, so
        the entire point is that this costs the same as `trunk`. An override
        that runs the trunk twice has given up the only reason to tap at all.

        **`f` must be bit-identical to `trunk(x)`.** The seam is what every
        cached feature, every baseline number and every frozen head was built
        against; a tap-emitting path that quietly returns a slightly different
        `f` would make the ladder's numbers incomparable with the plain
        adapter's. `verify_taps` asserts it.
        """
        if self.taps():
            raise NotImplementedError(
                f"{type(self).__name__}.taps() names {len(self.taps())} tap(s) but "
                f"trunk_with_taps() is not implemented, so nothing can read them."
            )
        return self.trunk(x), None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def preprocess_fn(self):
        """Delegate: the cache must use the detector's own transform, unchanged.

        A split that alters preprocessing silently invalidates the cache against
        the harness it will be evaluated in.
        """
        return self.detector.preprocess_fn()

    def assert_frozen(self) -> None:
        """Called every step of training, not once at startup.

        A BatchNorm-containing detector left in train mode updates its running
        statistics on degraded data and adapts itself, contaminating the exact
        comparison being made -- and anything can call `.train()` on a parent
        module between one step and the next.
        """
        if self.detector.training:
            raise RuntimeError(
                f"{self.name} is in training mode; the trunk and head are frozen. "
                "Call .eval() on the split before training the adapter."
            )
        trainable = [n for n, p in self.detector.named_parameters() if p.requires_grad]
        if trainable:
            raise RuntimeError(
                f"{self.name} has {len(trainable)} trainable parameter(s) "
                f"{trainable[:5]}; only the adapter may train."
            )
