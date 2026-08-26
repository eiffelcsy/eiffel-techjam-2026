"""Both stages, end to end, on a real rendered cache.

Small and fast, but not a mock: this renders features to disk, trains an adapter
against them, checkpoints it, reloads it, and trains the discrepancy head against
the frozen result. It is the test that would have caught every wiring mistake in
`loop.py`, and it is the shape of the `max_images: 64, n_epochs: 2` smoke run the
README prescribes before a real one.
"""

import json

import pytest
import torch

from grace.cache.reader import FeatureCache
from grace.cache.schedule import EpochSchedule, val_epochs
from grace.cache.spec import CacheSpec, sha_manifest, sha_preprocess
from grace.cache.writer import build_cache
from grace.config import (
    AdapterConfig, DiscrepancyConfig, DiscrepancyTrainConfig, LossConfig,
    SamplingConfig, TrainConfig,
)
from grace.models.factory import load_adapter
from grace.train.loop import train_adapter, train_discrepancy
from pipeline.degrade.conditions import load_grid
from tests.fixtures import SPECS, MLPHead, ToySplit, write_images

N_IMAGES = 32
GRID_FILE = "../eval_pipeline/configs/degradations.yaml"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    manifest = write_images(root / "images", N_IMAGES)
    spec = SPECS["vector"]
    # A nonlinear head on purpose: it is what makes posterior sampling and the
    # Jacobian weighting non-trivial, and a linear one would hide bugs in both.
    split = ToySplit(spec, head=MLPHead(spec))
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)

    cache_spec = CacheSpec(
        detector="toy", feature=spec, n=len(manifest), shard_size=16,
        manifest_sha=sha_manifest(manifest), schedule_sha=schedule.fingerprint(),
        detector_sha="toy", preprocess_sha=sha_preprocess(split.preprocess_fn()),
    )
    cache_dir = root / "cache"
    build_cache(
        split, manifest, cache_dir, cache_spec, schedule,
        [0, 1, *val_epochs(1)], batch_size=8, num_workers=0,
    )
    return root, manifest, split, schedule, cache_dir


def _train_cfg(root, cache_dir, **kw):
    return TrainConfig(
        run_id=kw.pop("run_id", "smoke"),
        cache_dir=str(cache_dir),
        epochs=2, batch_size=8, warmup_steps=1, num_workers=0,
        out_dir=str(root / "ckpt"),
        adapter=AdapterConfig(bottleneck=8, n_blocks=1, **kw.pop("adapter", {})),
        sampling=SamplingConfig(k_train=2, k_eval=2),
        loss=LossConfig(**kw.pop("loss", {})),
        **kw,
    )


def test_stage_one_runs_and_checkpoints(workspace):
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="stage1")
    summary = train_adapter(cfg, split, manifest, schedule)

    out = root / "ckpt" / "stage1"
    assert summary["steps"] > 0
    assert (out / "last.pt").exists() and (out / "ema.pt").exists()
    assert json.loads((out / "summary.json").read_text())["target_view"] == "clean"
    # validation ran on the held-out degradation epoch, not a training one
    assert f"epoch_{min(val_epochs(1))}" in summary["validation"]


def test_stage_one_actually_moves_the_adapter(workspace):
    """A run that leaves the adapter at its identity init has trained nothing --
    the gate should lift off sigmoid(-4) and the correction become nonzero."""
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="moved")
    train_adapter(cfg, split, manifest, schedule)

    adapter = load_adapter(root / "ckpt" / "moved" / "last.pt", split.feature_spec)
    f = torch.randn(4, *split.feature_spec.shape)
    assert not torch.allclose(adapter(f), f, atol=1e-6)


def test_diagnostics_are_logged(workspace):
    """`cos(Δ, j)` from the first step -- it is the figure, not an afterthought."""
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="diag")
    summary = train_adapter(cfg, split, manifest, schedule)
    first = summary["history"][0]
    assert "cos_decision" in first and "gate" in first
    assert 0.0 <= first["cos_decision"] <= 1.0


def test_control_arm_uses_the_degraded_target(workspace):
    """Arm A must run through the same code path as arm B, differing in one key."""
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="armA", target_view="degraded")
    assert train_adapter(cfg, split, manifest, schedule)["target_view"] == "degraded"


def test_bad_target_view_is_rejected(workspace):
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="bad", target_view="nonsense")
    with pytest.raises(ValueError, match="target_view"):
        train_adapter(cfg, split, manifest, schedule)


@pytest.mark.parametrize("weighting", ["jacobian", "none"])
def test_both_weightings_train(workspace, weighting):
    """The plain-MSE ablation must be one config key away, and must run."""
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id=f"w_{weighting}",
                     loss={"weighting": weighting})
    assert train_adapter(cfg, split, manifest, schedule)["steps"] > 0


def test_stochastic_adapter_trains_and_reports_spread(workspace):
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="noisy", adapter={"noise_dim": 4})
    summary = train_adapter(cfg, split, manifest, schedule)
    held = summary["validation"][f"epoch_{min(val_epochs(1))}"]
    assert "posterior_spread" in held


def test_stage_two_trains_against_a_frozen_adapter(workspace):
    """The supervised variant, and the E4 measurement it exists to enable."""
    root, manifest, split, schedule, cache_dir = workspace
    stage1 = _train_cfg(root, cache_dir, run_id="for_disc")
    train_adapter(stage1, split, manifest, schedule)
    checkpoint = root / "ckpt" / "for_disc" / "ema.pt"
    before = load_adapter(checkpoint, split.feature_spec).state_dict()

    cfg = DiscrepancyTrainConfig(
        run_id="disc", cache_dir=str(cache_dir), adapter_checkpoint=str(checkpoint),
        epochs=1, batch_size=8, num_workers=0, out_dir=str(root / "ckpt"),
        discrepancy=DiscrepancyConfig(hidden=8, proj=4),
    )
    summary = train_discrepancy(cfg, split, manifest)

    assert (root / "ckpt" / "disc" / "discrepancy.pt").exists()
    row = next(iter(summary["validation"].values()))
    assert {"auc_main", "auc_aux", "auc_fused"} <= set(row)

    # The claim that makes GRACE and GRACE-D comparable: stage 2 leaves the
    # adapter bit-identical, so the label-free claim survives the supervised
    # variant.
    after = load_adapter(checkpoint, split.feature_spec).state_dict()
    for key in before:
        assert torch.equal(before[key], after[key]), key
