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
    TrainConfig,
)
from grace.cache.reader import FeatureCache
from grace.models.factory import build_adapter, load_adapter
from grace.train.loop import train_adapter, train_discrepancy, validate
from pipeline.degrade.conditions import load_grid
from tests.fixtures import SPECS, MLPHead, ToySplit, write_images

N_IMAGES = 32
GRID_FILE = "../eval_pipeline/configs/degradations.yaml"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    manifest = write_images(root / "images", N_IMAGES)
    spec = SPECS["vector"]
    # A nonlinear head on purpose: it is what makes the Jacobian weighting and
    # head_kl non-trivial, and a linear one would hide bugs in both.
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
    assert (
        f"epoch_{min(val_epochs(1))}" in summary["validation"]["held_out_degradations"]
    )


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


def test_val_every_records_a_row_per_validated_epoch(workspace):
    """The mid-run curve. `val_history` is additive: `validation` still holds
    the finished adapter's numbers, so nothing downstream has to ask which
    schedule a run used to find the number it wants."""
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="valevery", val_every=1)
    summary = train_adapter(cfg, split, manifest, schedule)

    rows = summary["val_history"]
    assert [r["epoch"] for r in rows] == summary["epochs"]
    assert all(r["step"] > 0 for r in rows)
    for row in rows:
        assert (
            f"epoch_{min(val_epochs(1))}" in row["held_out_degradations"]
        )
    assert "held_out_degradations" in summary["validation"]


def test_val_every_off_by_default(workspace):
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="noval")
    assert train_adapter(cfg, split, manifest, schedule)["val_history"] == []


def test_val_every_does_not_perturb_training(workspace):
    """Every DataLoader draws a base seed from the global generator, so an
    unforked mid-run validation would shift the training loader's own stream.
    Two runs at the same seed differing only in `val_every` must end
    bit-identical."""
    root, manifest, split, schedule, cache_dir = workspace
    for run_id, every in (("rng_off", 0), ("rng_on", 1)):
        cfg = _train_cfg(root, cache_dir, run_id=run_id, val_every=every)
        train_adapter(cfg, split, manifest, schedule)

    spec = split.feature_spec
    off = load_adapter(root / "ckpt" / "rng_off" / "last.pt", spec).state_dict()
    on = load_adapter(root / "ckpt" / "rng_on" / "last.pt", spec).state_dict()
    for key in off:
        assert torch.equal(off[key], on[key]), key


def test_image_axis_scores_only_the_current_epoch(workspace):
    """The held-out IMAGE caches carry every rendered epoch, but a mid-run pass
    wants one draw, not a sweep over all of them -- on the real PoC caches that
    is 14 passes per val set per validation. The held-out DEGRADATION axis is a
    different question and is always scored in full."""
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="axes")
    cache = FeatureCache(cache_dir)
    adapter = build_adapter(split.feature_spec, cfg.adapter)
    val_sets = [("toy", cache, manifest)]
    held = {f"epoch_{e}" for e in val_epochs(1)}

    mid = validate(cfg, adapter, split, cache, manifest, None, val_sets, epoch=1)
    assert list(mid["held_out_images/toy"]) == ["epoch_1"]
    assert set(mid["held_out_degradations"]) == held

    # epoch=None is the end-of-run pass: the full sweep, paid once, so the
    # `validation` block stays comparable to every summary written before this.
    final = validate(cfg, adapter, split, cache, manifest, None, val_sets)
    assert set(final["held_out_images/toy"]) == {"epoch_0", "epoch_1", *held}
    assert set(final["held_out_degradations"]) == held


def test_unrendered_epoch_notes_rather_than_crashes(workspace):
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="missing")
    cache = FeatureCache(cache_dir)
    adapter = build_adapter(split.feature_spec, cfg.adapter)
    out = validate(cfg, adapter, split, cache, manifest, None,
                   [("toy", cache, manifest)], epoch=99)
    assert "note" in out["held_out_images/toy"]


def test_validation_reports_detection_metrics(workspace):
    """AUC/accuracy through the frozen head, for all three views. The adapted
    number is meaningless without the degraded one it must beat and the clean
    one it is bounded by, so all three are asserted together."""
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="detect")
    summary = train_adapter(cfg, split, manifest, schedule)
    row = summary["validation"]["held_out_degradations"][f"epoch_{min(val_epochs(1))}"]

    for view in ("degraded", "adapted", "clean"):
        assert 0.0 <= row[f"auc_{view}"] <= 1.0, view
        assert 0.0 <= row[f"acc_{view}"] <= 1.0, view
        assert 0.0 <= row[f"f1_{view}"] <= 1.0, view
    assert "retention" in row and "threshold" in row
    # The alignment metrics did not get displaced by the detection ones.
    assert "cosine_to_clean" in row and "gate" in row


def test_accuracy_uses_one_threshold_across_views(workspace):
    """The harness rule: pick the operating point on clean, apply it unchanged.
    A per-view threshold would hide exactly the calibration drift these numbers
    exist to expose."""
    root, manifest, split, schedule, cache_dir = workspace
    cfg = _train_cfg(root, cache_dir, run_id="thr")
    cache = FeatureCache(cache_dir)
    adapter = build_adapter(split.feature_spec, cfg.adapter)
    out = validate(cfg, adapter, split, cache, manifest)
    row = out["held_out_degradations"][f"epoch_{min(val_epochs(1))}"]

    from pipeline.eval.metrics import threshold_from_clean
    from grace.train.data import build_loader

    scores, labels = [], []
    for batch in build_loader(cfg, cache, manifest, None, min(val_epochs(1)),
                              shuffle=False):
        scores.append(split.head(batch["f_clean"].float()).detach().numpy())
        labels.append(batch["label"].numpy())
    import numpy as np
    expected = threshold_from_clean(np.concatenate(scores), np.concatenate(labels))
    assert row["threshold"] == pytest.approx(expected)


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
    # `validation` is keyed by AXIS first (`held_out_degradations`, and one
    # `held_out_images/<name>` per val set), then by epoch -- the same shape
    # stage 1 writes, so an E4 sweep and the retention curve share an axis.
    assert "held_out_degradations" in summary["validation"]
    row = next(iter(summary["validation"]["held_out_degradations"].values()))
    assert {"auc_main", "auc_aux", "auc_fused"} <= set(row)

    # The claim that makes GRACE and GRACE-D comparable: stage 2 leaves the
    # adapter bit-identical, so the label-free claim survives the supervised
    # variant.
    after = load_adapter(checkpoint, split.feature_spec).state_dict()
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_decay_gate_false_exempts_only_the_gate_logits():
    """The exemption has to be exactly the gate, or the arm is not attributable.

    `decay_gate: false` is read against a control that differs in nothing else,
    so every other parameter must stay in the decayed group -- including the
    severity head, which is optimized alongside the adapter.
    """
    from grace.models.factory import build_severity_head
    from grace.train.loop import _param_groups
    from tests.fixtures import SPECS

    spec = SPECS["vector"]
    adapter = build_adapter(spec, AdapterConfig())
    sev = build_severity_head(spec)
    n_params = len(list(adapter.parameters())) + len(list(sev.parameters()))

    on = _param_groups(adapter, sev, TrainConfig(run_id="t", cache_dir="x"))
    assert len(on) == 1 and len(on[0]["params"]) == n_params
    assert "weight_decay" not in on[0]

    off = _param_groups(
        adapter, sev, TrainConfig(run_id="t", cache_dir="x", decay_gate=False)
    )
    decayed, exempt = off
    assert exempt["weight_decay"] == 0.0
    assert [id(p) for p in exempt["params"]] == [id(adapter.gate_logit)]
    assert len(decayed["params"]) == n_params - 1
    assert "weight_decay" not in decayed

