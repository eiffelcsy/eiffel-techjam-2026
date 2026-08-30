"""Stage 2 for the frequency branch, end to end, on a real rendered cache.

Small and fast, but not a mock: this renders features AND the DCT view to disk,
trains an adapter against them, freezes it, trains the enricher against the
frozen result, checkpoints it and reloads it. It is the test that would catch a
wiring mistake in `train_enrich` -- and the one thing a unit test on
`FrequencyEnricher` cannot catch is the enricher being spliced at the wrong
point, which is exactly what `validation.step_0` is for.

Mirrors `test_train_smoke.py` deliberately: same shape, same fixtures, one stage
further on.
"""

import json

import pytest
import torch

from grace.cache.schedule import EpochSchedule, val_epochs
from grace.cache.spec import CacheSpec, sha_manifest, sha_preprocess
from grace.cache.writer import build_cache
from grace.config import (
    AdapterConfig, EnricherConfig, EnrichTrainConfig, FreqConfig, LossConfig,
    TrainConfig,
)
from grace.models.factory import load_enricher
from grace.train.loop import train_adapter, train_enrich
from pipeline.degrade.conditions import load_grid
from tests.fixtures import SPECS, MLPHead, ToySplit, write_images

N_IMAGES = 32
GRID_FILE = "../eval_pipeline/configs/degradations.yaml"
FREQ = FreqConfig(enabled=True, patch=2, grid=3)   # (9, 12): tiny, same structure


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("enrich")
    manifest = write_images(root / "images", N_IMAGES)
    spec = SPECS["vector"]
    # A nonlinear head, as stage 1's smoke test uses: the BCE is taken through
    # it, and a linear one would hide a gradient that never reaches the enricher.
    split = ToySplit(spec, head=MLPHead(spec))
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)

    cache_spec = CacheSpec(
        detector="toy", feature=spec, n=len(manifest), shard_size=16,
        manifest_sha=sha_manifest(manifest), schedule_sha=schedule.fingerprint(),
        detector_sha="toy", preprocess_sha=sha_preprocess(split.preprocess_fn()),
        freq_feature=FREQ.feature(), freq_sha=FREQ.fingerprint(),
    )
    cache_dir = root / "cache"
    build_cache(
        split, manifest, cache_dir, cache_spec, schedule,
        [0, 1, *val_epochs(1)], batch_size=8, num_workers=0, freq=FREQ.build(),
    )

    stage1 = TrainConfig(
        run_id="stage1", cache_dir=str(cache_dir), epochs=2, batch_size=8,
        warmup_steps=1, num_workers=0, out_dir=str(root / "ckpt"),
        adapter=AdapterConfig(bottleneck=8, n_blocks=1), loss=LossConfig(),
    )
    train_adapter(stage1, split, manifest, schedule)
    return root, manifest, split, root / "ckpt" / "stage1" / "ema.pt", cache_dir


def _cfg(root, cache_dir, adapter, **kw):
    return EnrichTrainConfig(
        run_id=kw.pop("run_id", "enrich"),
        cache_dir=str(cache_dir),
        adapter_checkpoint=str(adapter),
        epochs=2, batch_size=8, num_workers=0, log_every=1,
        out_dir=str(root / "ckpt"),
        enricher=EnricherConfig(d_model=16, n_heads=2, **kw.pop("enricher", {})),
        freq=FREQ,
        **kw,
    )


def test_stage_two_runs_and_checkpoints(workspace):
    root, manifest, split, adapter, cache_dir = workspace
    summary = train_enrich(_cfg(root, cache_dir, adapter, run_id="e"), split, manifest)

    out = root / "ckpt" / "e"
    assert (out / "enricher.pt").exists()
    assert json.loads((out / "summary.json").read_text())["finetune_head"] is False
    assert summary["validation"]["held_out_degradations"]
    assert len(summary["gates"]) == 2


def test_the_identity_holds_at_step_zero(workspace):
    """E10, measured inside the training run rather than asserted.

    Before the first optimizer step every expert's output projection is still
    zero, so `fused == f_corrected` bit for bit and the two AUC columns must be
    EQUAL -- not close. A unit test on the module cannot catch the enricher being
    spliced at the wrong point; this can.
    """
    root, manifest, split, adapter, cache_dir = workspace
    summary = train_enrich(_cfg(root, cache_dir, adapter, run_id="id"), split, manifest)

    rows = summary["validation"]["step_0"]
    assert rows, "step 0 validation scored nothing"
    for name, row in rows.items():
        assert row["auc_corrected"] == row["auc_fused"], name
        assert row["enrichment_norm"] == 0.0, name


def test_training_actually_moves_the_enricher(workspace):
    """A run that leaves it at the identity has trained nothing, and every later
    comparison would be a model against itself."""
    root, manifest, split, adapter, cache_dir = workspace
    summary = train_enrich(_cfg(root, cache_dir, adapter, run_id="moved"), split, manifest)

    after = summary["validation"]["held_out_degradations"]
    assert any(row["enrichment_norm"] > 0 for row in after.values())
    enricher = load_enricher(str(root / "ckpt" / "moved" / "enricher.pt"))
    assert any(
        float(expert.out.weight.detach().abs().sum()) > 0
        for expert in enricher.experts
    )


def test_the_adapter_is_not_touched(workspace):
    """GRACE and GRACE-freq must ship the same adapter weights, bit for bit --
    which is what keeps "the adapter is trained without labels" literally true of
    the artifact, not just of a stage."""
    root, manifest, split, adapter, cache_dir = workspace
    before = torch.load(adapter, map_location="cpu", weights_only=False)["state_dict"]
    train_enrich(_cfg(root, cache_dir, adapter, run_id="frozen"), split, manifest)
    after = torch.load(adapter, map_location="cpu", weights_only=False)["state_dict"]
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_the_finetuned_head_arm_carries_its_head(workspace):
    """E14. The head travels inside the enricher checkpoint because the two are
    only meaningful together: a config naming one without the other would score a
    head against features it never saw."""
    root, manifest, split, adapter, cache_dir = workspace
    frozen_head = [p.detach().clone() for p in split.detector.head.parameters()]
    summary = train_enrich(
        _cfg(root, cache_dir, adapter, run_id="ft", finetune_head=True), split, manifest
    )
    assert summary["finetune_head"] is True

    payload = torch.load(
        root / "ckpt" / "ft" / "enricher.pt", map_location="cpu", weights_only=False
    )
    assert payload["finetune_head"] is True and payload["head_state_dict"] is not None
    # The split's own head is the frozen reference every other arm is scored
    # with, so the fine-tuned arm must train a COPY and leave it alone.
    for before, param in zip(frozen_head, split.detector.head.parameters()):
        assert torch.equal(before, param.detach())


def test_an_unanchored_arm_runs(workspace):
    """E15. `lam_anchor: 0` is a shipped arm, so it has to be a value the loop
    handles rather than a division that only ever ran positive."""
    root, manifest, split, adapter, cache_dir = workspace
    summary = train_enrich(
        _cfg(root, cache_dir, adapter, run_id="noanchor", lam_anchor=0.0),
        split, manifest,
    )
    assert summary["validation"]["held_out_degradations"]


@pytest.mark.parametrize(
    "kwargs", [{"n_bands": 1}, {"top_k": 2}, {"pos_emb": False}],
    ids=["e11-single", "e13-topk", "e13-nopos"],
)
def test_every_enricher_ablation_trains(workspace, kwargs):
    """Each is a shipped arm reachable from a CLI flag, and each has to survive a
    real training step -- not merely build."""
    root, manifest, split, adapter, cache_dir = workspace
    run = "abl_" + "_".join(f"{k}{v}" for k, v in kwargs.items())
    summary = train_enrich(
        _cfg(root, cache_dir, adapter, run_id=run, enricher=kwargs), split, manifest
    )
    assert len(summary["gates"]) == kwargs.get("n_bands", 2)


def test_a_cache_without_the_freq_view_is_refused(workspace, tmp_path):
    """The refusal has to arrive at startup. Discovered per batch it would be a
    crash minutes into a run; not discovered at all it would be an enricher
    trained over zeros, reported as a null result."""
    root, manifest, split, adapter, _ = workspace
    plain_dir = tmp_path / "plain"
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    spec = SPECS["vector"]
    build_cache(
        split, manifest, plain_dir,
        CacheSpec(detector="toy", feature=spec, n=len(manifest), shard_size=16),
        schedule, [0], batch_size=8, num_workers=0,
    )
    with pytest.raises(FileNotFoundError, match="without a frequency view"):
        train_enrich(
            _cfg(root, plain_dir, adapter, run_id="nofreq"), split, manifest
        )


def test_a_different_dct_protocol_is_refused(workspace):
    """Same shape, different frequencies. The enricher's band masks are indexed
    by position along the coefficient axis, so this loads and means something
    else."""
    root, manifest, split, adapter, cache_dir = workspace
    cfg = _cfg(root, cache_dir, adapter, run_id="badproto")
    cfg.freq = FreqConfig(enabled=True, patch=2, grid=3, radial=False)
    with pytest.raises(ValueError, match="freq_sha differs"):
        train_enrich(cfg, split, manifest)
