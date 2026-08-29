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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from grace.splits.base import FeatureSpec

CLEAN_VIEW = "clean"
"""The epoch-independent view. Degraded views are named `epoch=%03d`."""

SPEC_FILE = "spec.json"
INDEX_FILE = "index.npy"
DONE_FILE = ".done"

TAP_DIR = "taps"
"""Tap views live in a subdirectory rather than beside the feature views.

A sibling naming like `epoch=000.taps` would be caught by
`FeatureCache.epochs()`'s `epoch=*` glob and parsed as an epoch number, which
fails on the `int()` -- and a scheme whose correctness depends on every future
glob remembering to exclude it is the wrong scheme. Nesting keeps the existing
directory layout untouched: a cache without taps is byte-identical to one
rendered before taps existed.
"""


def view_name(epoch: int | None) -> str:
    return CLEAN_VIEW if epoch is None else f"epoch={epoch:03d}"


def tap_view_name(epoch: int | None) -> str:
    """The tap view paired with `view_name(epoch)` -- `taps/clean`, `taps/epoch=007`.

    Clean taps are rendered alongside the degraded ones so that the two views
    of a row stay a single lookup at the same offset. Nothing in the objective
    reads them today; `FeatureCache.clean_taps` is what makes them available to
    analysis scripts without a re-render.
    """
    return f"{TAP_DIR}/{view_name(epoch)}"


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
    taps: tuple[str, ...] = field(default_factory=tuple)
    """Names of the intermediate taps rendered alongside the features, in group
    order -- `("block00", "block02", ...)`, straight from `SplitDetector.taps()`.

    Empty means a cache with no tap views, which is every cache rendered before
    the ladder existed and every cache for a run that does not want one. The
    field was reserved from the first render for exactly this: enabling taps
    *adds views* to the directory layout rather than changing the on-disk format
    of what is already there.
    """
    tap_feature: FeatureSpec | None = None
    """Shape of one image's stacked taps, `(K, tap_dim)`. Set iff `taps` is."""

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
            "taps": list(self.taps),
            "tap_feature": self.tap_feature.to_dict() if self.tap_feature else None,
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
            taps=tuple(d.get("taps", ())),
            tap_feature=(
                FeatureSpec.from_dict(d["tap_feature"]) if d.get("tap_feature") else None
            ),
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
        # Taps are checked only when the *reader* wants them, and the rule is
        # SUBSET rather than equality: a cache rendering five blocks can serve a
        # run that reads four of them, because dropping a tap is a column
        # selection at read time and needs no re-render. Adding one does -- the
        # bytes are simply not there -- so a requested tap the cache does not
        # hold is still a hard refusal.
        #
        # A plain run against a cache that happens to carry taps is fine and
        # ignores them, which is why this is gated on `other.taps`.
        if other.taps:
            missing = [t for t in other.taps if t not in self.taps]
            if missing:
                raise ValueError(
                    f"tap mismatch: this split wants {list(other.taps)} but the "
                    f"cache holds {list(self.taps) or 'no taps'} -- {missing} "
                    f"was never rendered. Taps can be dropped at read time but "
                    f"not conjured: re-render with scripts/build_cache.py, or "
                    f"align `split_args.tap_blocks` with the cache."
                )
            if (
                self.tap_feature is not None
                and other.tap_feature is not None
                and self.tap_feature.dim != other.tap_feature.dim
            ):
                raise ValueError(
                    f"tap width mismatch: cache holds {self.tap_feature.dim}-d "
                    f"taps, this split emits {other.tap_feature.dim}-d. Selecting "
                    f"a subset changes how many taps are read, never how wide "
                    f"they are -- this is a different detector or pooling."
                )

    def bytes_per_view(self) -> int:
        """One image, one view -- features plus taps if this cache carries them."""
        tap = self.tap_feature.bytes_per_image() if self.tap_feature else 0
        return self.feature.bytes_per_image() + tap

    def nbytes(self, n_views: int | None = None) -> int:
        """Total on-disk size. Printed by `build_cache.py --dry-run` before
        committing to a multi-hour render.

        Taps count: they are the reason to run `--dry-run` at all now, since
        five of them multiply a DINOv3 cache by six.
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
