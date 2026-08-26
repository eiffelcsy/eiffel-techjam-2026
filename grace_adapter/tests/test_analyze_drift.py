"""E0's two easy ways to be wrong: the cache it needs, and the device it runs on.

`analyze_drift.py`'s whole analysis is a comparison of the `clean` view against a
degraded one, so there is nothing for it to compute until `build_cache.py` has
run. The README numbered it "0" for years because it comes before anything is
*trained* -- which is true, and which reads as "before the render" to anyone
following the list top to bottom.

Pinned here, so each is enforced by the code rather than only asserted by a
comment:

  * no cache directory at all -> fails opening it, in the first statement of main
  * a cache with the clean view but no finalized degraded view -> the explicit
    `no rendered epochs under <dir>` exit, which is the state a `build_cache.py`
    run interrupted before its first epoch leaves behind
  * the split is built ONCE, not once per batch -- `build_detector` loads weights
    from disk, and the head Jacobian does not change between batches
  * the head gradient crosses devices -- the cache is memmapped numpy and hands
    back CPU tensors, while `device: auto` puts the detector on MPS or CUDA
"""

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest
import torch

from grace.cache.reader import FeatureCache
from grace.cache.schedule import EpochSchedule
from grace.cache.spec import CacheSpec, sha_manifest, sha_preprocess
from grace.cache.writer import build_cache
from pipeline.degrade.conditions import load_grid
from tests.fixtures import SPECS, ToySplit, features, write_images

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_drift.py"
GRID_FILE = "../eval_pipeline/configs/degradations.yaml"


def _module():
    """Import the script as a module, to reach its helpers directly.

    `scripts/` is not a package -- the entry points are meant to be *run*, not
    imported -- so the two functions worth unit-testing are reached this way
    rather than by making the directory importable for the sake of a test.
    """
    spec = importlib.util.spec_from_file_location("analyze_drift", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(argv):
    """Execute the script as `__main__` with a patched argv."""
    saved = sys.argv
    sys.argv = ["analyze_drift.py", *argv]
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = saved


def _dataset_yaml(tmp_path, manifest):
    """A minimal dataset config pointing at a manifest written next to it."""
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path)
    path = tmp_path / "dataset.yaml"
    path.write_text(f"name: toy\nmanifest: {manifest_path}\n", encoding="utf-8")
    return path


def _build(tmp_path, epochs, name="cache"):
    """Render a toy cache with the given epochs. `[]` = clean view only."""
    manifest = write_images(tmp_path / f"images_{name}", 8)
    spec = SPECS["vector"]
    split = ToySplit(spec)
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    root = tmp_path / name
    build_cache(
        split, manifest, root,
        CacheSpec(
            detector="toy", feature=spec, n=len(manifest), shard_size=8,
            manifest_sha=sha_manifest(manifest), schedule_sha=schedule.fingerprint(),
            detector_sha="toy", preprocess_sha=sha_preprocess(split.preprocess_fn()),
        ),
        schedule, epochs, batch_size=4, num_workers=0,
    )
    return root, manifest


def _rendered_cache(tmp_path):
    """A usable cache: clean view plus one degraded epoch."""
    return _build(tmp_path, [0], name="rendered")


@pytest.fixture
def clean_only_cache(tmp_path):
    """What an interrupted render leaves: the clean view, no degraded epoch."""
    return _build(tmp_path, [], name="clean_only")


def test_no_cache_at_all_fails_before_anything_else(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        _run(["--cache", str(tmp_path / "never-rendered"),
              "--dataset", "../eval_pipeline/configs/datasets/ntire_train.yaml"])


def test_clean_view_without_a_degraded_epoch_exits_with_the_reason(clean_only_cache, tmp_path):
    """The drift comparison needs both sides. A clean-only cache is a render that
    was interrupted, not a cache E0 can work with."""
    root, manifest = clean_only_cache
    assert FeatureCache(root).epochs() == ()

    dataset = _dataset_yaml(tmp_path, manifest)
    with pytest.raises(SystemExit, match="no rendered epochs"):
        _run(["--cache", str(root), "--dataset", str(dataset)])


def test_the_split_is_built_once_not_once_per_batch(clean_only_cache, tmp_path, monkeypatch):
    """`build_detector` loads weights from disk and runs `verify_split`.

    Building it inside the batch loop reloads the entire trunk for every batch --
    dozens of times over a multi-epoch cache -- to recompute a gradient that
    depends only on the frozen head. Cheap to get wrong, invisible except as
    "why is E0 slow", and this counts the calls.
    """
    import pipeline.config
    import pipeline.detectors

    root, manifest = _rendered_cache(tmp_path)
    spec = SPECS["vector"]
    split = ToySplit(spec)

    calls = []
    monkeypatch.setattr(pipeline.config, "load_detector_config", lambda e: {"stub": True})
    monkeypatch.setattr(
        pipeline.detectors, "build_detector",
        lambda cfg: (calls.append(1), split.detector)[1],
    )
    monkeypatch.setattr("grace.splits.build_split", lambda d, t, **kw: split)

    dataset = _dataset_yaml(tmp_path, manifest)
    _run([
        "--cache", str(root), "--dataset", str(dataset),
        "--detector", "stub.yaml", "--split", "stub.Split",
        "--batch-size", "2",                      # 8 images -> 4 batches
        "--out", str(tmp_path / "drift.json"),
    ])
    assert len(calls) == 1, f"detector built {len(calls)} times; must be once"


def test_head_gradient_comes_back_on_the_features_device():
    """The contract `main` relies on: everything else in the script stays on the
    CPU where the cache put it, so `j` must too."""
    spec = SPECS["vector"]
    mod = _module()
    split = ToySplit(spec)
    f = features(spec, batch=4)

    j = mod.head_gradient_for(split, f)
    assert j.shape == f.shape and j.device == f.device


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="needs a non-CPU device to cross",
)
def test_head_gradient_crosses_from_cpu_features_to_an_accelerator_head():
    """The exact failure: `RuntimeError: Passed CPU tensor to MPS op`.

    `FeatureCache` is memmapped numpy, so its tensors are always CPU; the
    detector goes wherever its config's `device:` put it, and `auto` is not CPU
    on any machine anyone trains on. The features move to the model and the
    gradient comes back.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    spec = SPECS["vector"]
    mod = _module()
    split = ToySplit(spec).to(device)
    f_cpu = features(spec, batch=4)              # as the cache hands it over

    j = mod.head_gradient_for(split, f_cpu)
    assert j.device == f_cpu.device == torch.device("cpu")
    assert torch.isfinite(j).all() and j.abs().sum() > 0
