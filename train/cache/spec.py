"""What a cache directory claims to be, and the hashes that prove it.

A feature cache is a pile of float16 with no provenance. Every failure mode is
silent -- misaligned rows, a rebuilt manifest, a nudged degradation parameter, a
detector loaded from different weights -- and each produces a plausible training
curve and meaningless results. So the spec carries a fingerprint for every input
the features depend on, and loading asserts all four:

    manifest_sha    row order and contents of the manifest (rows are the index)
    schedule_sha    grid + level weights + seed  (degraded views only)
    detector_sha    detector target + args, i.e. which weights produced these
    preprocess_sha  the detector's transform; a stochastic preprocess would make
                    the clean cache non-reproducible and is rejected here

Cheap to compute, once, and they turn the nastiest class of bug in the project
into an assertion at startup that names *which* input moved.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.splits.base import FeatureSpec

CLEAN_VIEW = "clean"
"""The epoch-independent view. Degraded views are named `epoch=%03d`."""

SPEC_FILE = "spec.json"
INDEX_FILE = "index.npy"
DONE_FILE = ".done"

FREQ_DIR = "freq"
"""The DCT side-view, nested in a subdirectory rather than beside the feature
views.

A sibling naming like `epoch=000.freq` would be caught by
`FeatureCache.epochs()`'s `epoch=*` glob and parsed as an epoch number, which
fails on the `int()`. Nesting keeps the existing directory layout untouched: a
cache without a frequency view is byte-identical to one rendered before the
frequency branch existed.
"""


def view_name(epoch: int | None) -> str:
    return CLEAN_VIEW if epoch is None else f"epoch={epoch:03d}"


def freq_view_name(epoch: int | None) -> str:
    """The frequency view paired with `view_name(epoch)` -- `freq/epoch=003`.

    Rendered for the clean view too, and it IS read: `analyze_freq.py` compares
    clean band energies against degraded ones, and it is the clean view that
    says what a band looks like before any transform touched it.
    """
    return f"{FREQ_DIR}/{view_name(epoch)}"


@dataclass(frozen=True)
class CacheSpec:
    """Serialized to `spec.json` at the root of a cache directory."""

    detector: str
    feature: FeatureSpec
    n: int                                     # rows per view
    views: tuple[str, ...] = ()
    shard_size: int = 50_000
    manifest_sha: str = ""
    schedule_sha: str = ""
    detector_sha: str = ""
    preprocess_sha: str = ""
    crop_sha: str = ""
    """Identity of the multi-scale window protocol -- `SampleCrop.fingerprint()`.

    Empty means no crop, which is every cache rendered before multi-scale
    training existed. That is not the same as "unknown": `preprocess_sha` cannot
    cover the crop because the crop happens in the *dataset*, before
    preprocessing, precisely so its randomness can be seeded on the image index
    without making the transform stochastic. Two caches over the same manifest,
    detector and schedule can hold features of entirely different windows of
    entirely different pixels, and this is the only field that says so.
    """
    freq_feature: FeatureSpec | None = None
    """Shape of one image's DCT view, `(n_cells, n_coeffs)` -- layout `tokens`.

    None means a cache with no frequency views, which is every cache rendered
    before the frequency branch existed. Presence is the decision: the field
    ADDS views to the directory layout and changes nothing about the bytes
    already there, so an old cache stays readable.
    """
    freq_sha: str = ""
    """Identity of the extraction protocol -- `FreqExtract.fingerprint()`.

    Unlike the view count, which is resumable (`build_cache` skips finished
    views, so epochs can be added later at zero rework), the coefficient set is
    not: changing the patch size, the cell grid or the radial ordering
    re-renders every frequency byte in the directory. So it is asserted on load
    beside the other fingerprints rather than trusted to match.
    """

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "feature": self.feature.to_dict(),
            "n": self.n,
            "views": list(self.views),
            "shard_size": self.shard_size,
            "manifest_sha": self.manifest_sha,
            "schedule_sha": self.schedule_sha,
            "detector_sha": self.detector_sha,
            "preprocess_sha": self.preprocess_sha,
            "crop_sha": self.crop_sha,
            "freq_feature": self.freq_feature.to_dict() if self.freq_feature else None,
            "freq_sha": self.freq_sha,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheSpec":
        return cls(
            detector=d["detector"],
            feature=FeatureSpec.from_dict(d["feature"]),
            n=d["n"],
            views=tuple(d.get("views", ())),
            shard_size=d.get("shard_size", 50_000),
            manifest_sha=d.get("manifest_sha", ""),
            schedule_sha=d.get("schedule_sha", ""),
            detector_sha=d.get("detector_sha", ""),
            preprocess_sha=d.get("preprocess_sha", ""),
            crop_sha=d.get("crop_sha", ""),
            freq_feature=(
                FeatureSpec.from_dict(d["freq_feature"]) if d.get("freq_feature") else None
            ),
            freq_sha=d.get("freq_sha", ""),
        )

    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / SPEC_FILE).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, root: str | Path) -> "CacheSpec":
        path = Path(root) / SPEC_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"no {SPEC_FILE} under {root} -- build the cache first "
                f"(scripts/build_cache.py)"
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def assert_compatible(self, other: "CacheSpec") -> None:
        """Raise naming the *specific* mismatched input, not just "cache invalid".

        `schedule_sha` is checked only when both sides declare one: the clean
        view does not depend on the schedule, so a run that reads clean features
        alone stays valid across a change to the degradation grid.
        """
        if (self.feature.layout, self.feature.shape) != (
            other.feature.layout,
            other.feature.shape,
        ):
            raise ValueError(
                f"feature layout mismatch: cache holds {self.feature.layout}"
                f"{self.feature.shape}, detector emits {other.feature.layout}"
                f"{other.feature.shape}"
            )
        checks = [
            ("manifest_sha", "the manifest was rebuilt or reordered"),
            ("detector_sha", "the detector config (weights or args) changed"),
            ("preprocess_sha", "the detector's preprocessing changed"),
            ("schedule_sha", "the degradation grid, level weights or seed changed"),
        ]
        for name, why in checks:
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine and theirs and mine != theirs:
                raise ValueError(
                    f"cache is stale: {name} differs ({mine} != {theirs}) -- {why}. "
                    f"Re-render with scripts/build_cache.py."
                )
        # `crop_sha` is compared strictly, empty included, and that is the one
        # place this method departs from the both-sides-declared rule above. For
        # the other four an empty string means "this side does not constrain it";
        # for the crop it means something specific -- WHOLE IMAGES, no window --
        # and reading a cache of 128-512px windows as though it held whole-image
        # features is exactly the silent mismatch worth refusing. The features
        # are of different pixels and nothing downstream would notice.
        if self.crop_sha != other.crop_sha:
            described = lambda s: s or "no crop (whole images)"
            raise ValueError(
                f"cache is stale: crop_sha differs ({described(self.crop_sha)} != "
                f"{described(other.crop_sha)}) -- the multi-scale window protocol "
                f"changed, so these are features of different windows. Re-render "
                f"with scripts/build_cache.py, or fix `crop:` in the run config."
            )

    def assert_freq_available(self, want: FeatureSpec, want_sha: str = "") -> None:
        """Refuse a cache that cannot serve the frequency branch.

        Split out of `assert_compatible` rather than folded into it because the
        two are asked by different callers at different times. Stage 1 and the
        eval arms read a cache that may or may not carry a frequency view and
        must not care; only stage 2's enrichment run needs one, and it needs it
        to be the same protocol its enricher was shaped for.

        Both halves are hard. A missing view is bytes that are simply not there.
        A protocol mismatch is worse than missing: the shapes can agree while the
        coefficients mean different frequencies, and the enricher's learned band
        masks are indexed by position along that axis.
        """
        if self.freq_feature is None:
            raise FileNotFoundError(
                f"this cache was rendered without a frequency view, so the "
                f"enricher has nothing to read. Re-render with `freq.enabled: "
                f"true` (configs/cache/wildfake_freq.yaml), or train the plain "
                f"adapter."
            )
        if (self.freq_feature.layout, self.freq_feature.shape) != (want.layout, want.shape):
            raise ValueError(
                f"frequency view mismatch: cache holds {self.freq_feature.layout}"
                f"{self.freq_feature.shape}, this run wants {want.layout}"
                f"{want.shape}. The coefficient set is a render-time commitment "
                f"-- re-render with scripts/build_cache.py."
            )
        if want_sha and self.freq_sha and self.freq_sha != want_sha:
            raise ValueError(
                f"cache is stale: freq_sha differs ({self.freq_sha} != {want_sha}) "
                f"-- the patch size, cell grid or radial ordering changed, so the "
                f"coefficient axis means something else. Same shape, different "
                f"frequencies. Re-render with scripts/build_cache.py."
            )

    def bytes_per_view(self) -> int:
        """One image, one view -- features plus the frequency view if it carries one."""
        freq = self.freq_feature.bytes_per_image() if self.freq_feature else 0
        return self.feature.bytes_per_image() + freq

    def nbytes(self, n_views: int | None = None) -> int:
        """Total on-disk size. Printed by `build_cache.py --dry-run` before
        committing to a multi-hour render.
        """
        views = n_views if n_views is not None else max(len(self.views), 1)
        return self.n * self.bytes_per_view() * views


def _blake(payload: str) -> str:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def sha_manifest(manifest) -> str:
    """Hash path, label and index in row order. Order is part of the identity."""
    rows = "|".join(
        f"{i}:{p}:{l}"
        for i, p, l in zip(manifest.index, manifest["path"], manifest["label"])
    )
    return _blake(rows)


def sha_detector(cfg) -> str:
    """Hash the DetectorConfig target and args."""
    return _blake(json.dumps({"target": cfg.target, "args": cfg.args}, sort_keys=True))


def sha_preprocess(preprocess, size: int = 64) -> str:
    """Hash a fixed probe image through the transform.

    Doubles as the determinism check: the transform is run twice and required to
    produce identical bytes. A detector whose preprocessing is stochastic (a
    random crop, say) cannot be cached, and should fail here rather than 40 GB
    later. DRCT is the known case -- force a deterministic center crop for cache
    building and evaluation, per the day-3 notes.
    """
    rng = np.random.default_rng(0)
    probe = Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8))
    first, second = preprocess(probe), preprocess(probe)
    if not torch.equal(first, second):
        raise ValueError(
            "preprocessing is not deterministic: the same image produced two "
            "different tensors. A stochastic transform cannot be cached -- pin "
            "its RNG or use a deterministic crop."
        )
    return _blake(_blake(str(first.shape)) + hashlib.blake2b(
        np.ascontiguousarray(first.float().numpy()).tobytes(), digest_size=8
    ).hexdigest())
