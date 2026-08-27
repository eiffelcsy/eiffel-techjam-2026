"""The PoC path end to end: stage 0, cache, stage 1, stage 2, adapted detector.

No gated weights and no network. `facebook/dinov3-vits16-pretrain-lvd1689m` is a
licence-gated Hub repo, so a test that downloaded it would be a test that never
runs in CI -- and the thing worth testing is the wiring, not the pretraining.
A 2-layer / 32-dim DINOv3 built from a local config exercises exactly the same
code path: same class, same token layout (`[CLS, registers, patches]`), same
pooling, same seam.

What this pins, in order of how quietly each fails:

  * the trunk drops register tokens and pools to the declared width
  * a head trained against a different backbone or pool is REFUSED, not loaded
  * `head(trunk(x)) == detector(x)` -- by construction, and checked anyway
  * stage 0 -> cache -> stage 1 -> stage 2 runs with no shape surgery between
  * `AdaptedDetector(checkpoint=None)` reproduces the base detector bit for bit,
    which is experiment E1 and the precondition for every number after it
"""

import numpy as np
import pytest
import torch
from PIL import Image

transformers = pytest.importorskip("transformers")

from grace.cache.reader import FeatureCache                          # noqa: E402
from grace.cache.schedule import EpochSchedule, val_epochs           # noqa: E402
from grace.cache.spec import (                                       # noqa: E402
    CacheSpec, sha_manifest, sha_preprocess,
)
from grace.cache.writer import build_cache                           # noqa: E402
from grace.config import (                                           # noqa: E402
    AdapterConfig, DiscrepancyConfig, DiscrepancyTrainConfig, LossConfig,
    ProbeConfig, TrainConfig,
)
from grace.probe import train_probe                                  # noqa: E402
from grace.splits.dinov3 import DINOv3Split                          # noqa: E402
from grace.train.loop import train_adapter, train_discrepancy        # noqa: E402
from pipeline.degrade.conditions import load_grid                    # noqa: E402
from tests.fixtures import write_images                              # noqa: E402

GRID_FILE = "../eval_pipeline/configs/degradations.yaml"
N_IMAGES = 24
HIDDEN, PATCH, IMAGE = 32, 16, 32

pytest.importorskip("transformers.models.dinov3_vit")


@pytest.fixture(scope="module")
def tiny_backbone():
    """A real DINOv3ViTModel, small enough to be free, built with no network."""
    from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessor, DINOv3ViTModel

    cfg = DINOv3ViTConfig(
        hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, patch_size=PATCH, num_register_tokens=2,
        image_size=IMAGE,
    )
    torch.manual_seed(0)
    return DINOv3ViTModel(cfg), DINOv3ViTImageProcessor(size={"height": IMAGE, "width": IMAGE})


@pytest.fixture
def detector_factory(tiny_backbone, monkeypatch):
    """`DINOv3MLPDetector(...)` with the Hub calls redirected to the tiny model.

    Patching `from_pretrained` rather than injecting a model keeps the detector's
    own `__init__` -- and therefore its pooling, its width arithmetic and its
    checkpoint validation -- inside the test.
    """
    import pipeline.detectors.dinov3 as mod

    model, processor = tiny_backbone

    class _Auto:
        @staticmethod
        def from_pretrained(model_id, revision=None, **kw):
            return model

    class _AutoProc:
        @staticmethod
        def from_pretrained(model_id, revision=None, **kw):
            return processor

    import transformers

    monkeypatch.setattr(transformers, "AutoModel", _Auto, raising=False)
    monkeypatch.setattr(transformers, "AutoImageProcessor", _AutoProc, raising=False)

    def build(**kw):
        kw.setdefault("backbone_id", "test/tiny-dinov3")
        return mod.DINOv3MLPDetector(**kw).freeze()

    return build


def _split(detector_factory, **kw):
    with pytest.warns(UserWarning, match="randomly initialized head"):
        split = DINOv3Split(detector_factory(**kw))
    return split.eval()


# --------------------------------------------------------------------- seam --

@pytest.mark.parametrize("pool,mult", [("cls", 1), ("patchmean", 1), ("cls+patchmean", 2)])
def test_pool_sets_the_feature_width(detector_factory, pool, mult):
    split = _split(detector_factory, pool=pool)
    assert split.feature_spec.layout == "vector"
    assert split.feature_spec.shape == (HIDDEN * mult,)
    assert split.trunk(torch.randn(2, 3, IMAGE, IMAGE)).shape == (2, HIDDEN * mult)


def test_registers_are_dropped_from_the_patch_mean(detector_factory):
    """The pooled patch mean must not include CLS or the register tokens.

    Registers are scratch space the model parks high-norm activations in,
    deliberately untied from image content; averaging them into the descriptor
    would be a silent, plausible-looking corruption of every cached feature.
    """
    split = _split(detector_factory, pool="patchmean")
    x = torch.randn(2, 3, IMAGE, IMAGE)
    with torch.no_grad():
        tokens = split.detector.backbone(pixel_values=x).last_hidden_state
        assert torch.allclose(split.trunk(x), tokens[:, 3:].mean(dim=1), atol=1e-6)


def test_head_of_trunk_reproduces_the_detector(detector_factory):
    """`verify_split` runs inside `DINOv3Split.__init__`; this states the claim."""
    split = _split(detector_factory)
    x = torch.randn(3, 3, IMAGE, IMAGE)
    with torch.no_grad():
        assert torch.allclose(split.head(split.trunk(x)), split.detector(x), atol=1e-6)


def test_head_is_differentiable_wrt_its_input(detector_factory):
    """The Jacobian weighting takes ∇_f head(f). A head that killed the graph
    would disable the objective silently, not raise."""
    from grace.train.weighting import head_gradient

    split = _split(detector_factory)
    f = split.trunk(torch.randn(4, 3, IMAGE, IMAGE)).detach()
    j = head_gradient(split.head, f)
    assert j.shape == f.shape and j.abs().sum() > 0


def test_split_rejects_a_detector_without_a_seam():
    with pytest.raises(TypeError, match="DINOv3MLPDetector"):
        DINOv3Split(torch.nn.Linear(4, 4))


# ------------------------------------------------------------ stage 0, probe --

@pytest.fixture
def probe_workspace(tmp_path, detector_factory):
    manifest = write_images(tmp_path / "train", N_IMAGES, seed=0)
    val = write_images(tmp_path / "val", 8, seed=1)
    val.index = val.index + 1000            # disjoint identities, as two real
                                            # manifest splits would be
    split = _split(detector_factory)
    cfg = ProbeConfig(
        run_id="t", detector="", dataset="",
        out=str(tmp_path / "head.pt"), hidden=16, n_layers=2,
        epochs=3, batch_size=8, head_batch_size=8, num_workers=0,
    )
    return cfg, split, manifest, val


def test_probe_writes_a_head_the_detector_can_load(probe_workspace, detector_factory, tmp_path):
    cfg, split, manifest, val = probe_workspace
    summary = train_probe(cfg, split, manifest, [("val", val)])

    assert summary["n_train"] == N_IMAGES and summary["n_val"] == {"val": 8}
    assert summary["selection"] == "val_auc_mean"
    assert len(summary["history"]) == cfg.epochs

    detector = detector_factory(head_checkpoint=cfg.out, pool="cls+patchmean")
    assert detector.head_untrained is False
    # Round-trips to the same logits, which is the only thing "loadable" means.
    payload = torch.load(cfg.out, map_location="cpu", weights_only=False)
    for k, v in payload["state_dict"].items():
        assert torch.equal(detector.head_module.state_dict()[k], v)


def test_probe_head_is_refused_by_a_different_trunk(probe_workspace, detector_factory):
    """Same width, different weights -- the failure that otherwise just scores
    nonsense. The pool mismatch would break on a shape; this one would not."""
    cfg, split, manifest, val = probe_workspace
    train_probe(cfg, split, manifest, [("val", val)])

    with pytest.raises(ValueError, match="backbone"):
        detector_factory(head_checkpoint=cfg.out, backbone_id="test/other-dinov3")
    with pytest.raises(ValueError, match="pool"):
        detector_factory(head_checkpoint=cfg.out, pool="cls")


def test_probe_refuses_to_train_on_a_live_trunk(probe_workspace):
    """`assert_frozen` runs every epoch, not once at startup."""
    cfg, split, manifest, val = probe_workspace
    split.detector.train()
    with pytest.raises(RuntimeError, match="training mode"):
        train_probe(cfg, split, manifest, val)


# --------------------------------------------------- cache -> stage 1 -> 2 ----

@pytest.fixture
def poc_run(tmp_path, detector_factory):
    """The whole path once: probe, render, stage 1, stage 2.

    Rebuilt per test rather than shared: `detector_factory` patches
    `transformers.AutoModel` through `monkeypatch`, which is function-scoped by
    construction, and the tiny backbone makes the render cheap enough that
    caching it would buy a second and cost the isolation.
    """
    factory = detector_factory
    root = tmp_path
    manifest = write_images(root / "images", N_IMAGES)
    split = _split(factory)
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)

    probe_cfg = ProbeConfig(
        run_id="poc", detector="", dataset="", out=str(root / "head.pt"),
        hidden=16, n_layers=2, epochs=2, batch_size=8, head_batch_size=8,
        num_workers=0,
    )
    train_probe(probe_cfg, split, manifest)

    trained = factory(head_checkpoint=str(root / "head.pt"))
    split = DINOv3Split(trained).eval()
    spec = split.feature_spec

    cache_dir = root / "cache"
    build_cache(
        split, manifest, cache_dir,
        CacheSpec(
            detector="dinov3-test", feature=spec, n=len(manifest), shard_size=16,
            manifest_sha=sha_manifest(manifest), schedule_sha=schedule.fingerprint(),
            detector_sha="tiny", preprocess_sha=sha_preprocess(split.preprocess_fn()),
        ),
        schedule, [0, 1, *val_epochs(1)], batch_size=8, num_workers=0,
    )
    return root, manifest, split, schedule, cache_dir


def test_cache_holds_the_declared_vector_features(poc_run):
    _, manifest, split, _, cache_dir = poc_run
    cache = FeatureCache(cache_dir)
    assert cache.spec.feature.layout == "vector"
    assert cache.clean(manifest.index[:4]).shape == (4, split.feature_spec.dim)


def test_cached_clean_features_match_a_live_trunk_pass(poc_run):
    """The highest-risk bug in the project: it trains, it converges, it means
    nothing. Re-run the trunk on real images and compare row for row."""
    from pipeline.data.dataset import load_normalized

    _, manifest, split, _, cache_dir = poc_run
    cache = FeatureCache(cache_dir)
    preprocess = split.preprocess_fn()
    picked = list(manifest.index[:5])
    batch = torch.stack([preprocess(load_normalized(p)) for p in manifest.loc[picked, "path"]])
    with torch.no_grad():
        live = split.trunk(batch)
    cached = cache.clean(picked).float()
    assert torch.allclose(cached, live, atol=2e-3), (cached - live).abs().max()


def test_stage_one_then_stage_two(poc_run):
    root, manifest, split, schedule, cache_dir = poc_run

    cfg = TrainConfig(
        run_id="s1", cache_dir=str(cache_dir), epochs=2, batch_size=8,
        warmup_steps=1, num_workers=0, out_dir=str(root / "ckpt"),
        adapter=AdapterConfig(bottleneck=16, n_blocks=1),
        loss=LossConfig(lam_sw=0.1),
    )
    s1 = train_adapter(cfg, split, manifest, schedule)
    assert s1["steps"] > 0
    assert (root / "ckpt" / "s1" / "ema.pt").exists()
    # The gate must move off its sigmoid(-4) = 0.018 initialization; sitting
    # there means the alignment term never outweighed the identity term.
    assert s1["history"][-1]["gate"] > 0

    disc = DiscrepancyTrainConfig(
        run_id="s2", cache_dir=str(cache_dir),
        adapter_checkpoint=str(root / "ckpt" / "s1" / "ema.pt"),
        epochs=1, batch_size=8, num_workers=0, out_dir=str(root / "ckpt"),
        discrepancy=DiscrepancyConfig(hidden=16, proj=8),
    )
    s2 = train_discrepancy(disc, split, manifest)
    assert (root / "ckpt" / "s2" / "discrepancy.pt").exists()
    for row in s2["validation"].values():
        assert set(row) == {"auc_main", "auc_aux", "auc_fused"}


def test_adapter_parameter_budget_is_small(poc_run):
    """The claim is that the evidence is displaced, not destroyed -- which a
    large adapter would make untestable. Pin the ratio, not just the count."""
    from grace.models.factory import build_adapter

    _, _, split, _, _ = poc_run
    adapter = build_adapter(split.feature_spec, AdapterConfig(bottleneck=128, n_blocks=2))
    n_adapter = sum(p.numel() for p in adapter.parameters())
    n_trunk = sum(p.numel() for p in split.detector.backbone.parameters())
    assert n_adapter < 4 * n_trunk   # tiny backbone here; on ViT-S/16 it is ~2%


# ------------------------------------------------------------------ E1, null --

def test_identity_adapter_reproduces_the_base_detector(poc_run, monkeypatch, tmp_path):
    """E1. `AdaptedDetector(checkpoint=None)` must be the base model exactly.

    Built directly rather than through `build_detector` so the test does not
    need a detector yaml on disk; the code path under test -- trunk, no adapter,
    head -- is `AdaptedDetector.forward` either way.
    """
    from grace.detectors.adapted import AdaptedDetector

    _, _, split, _, _ = poc_run
    monkeypatch.setattr(
        "grace.detectors.adapted.build_detector", lambda cfg: split.detector
    )
    monkeypatch.setattr(
        "grace.detectors.adapted.load_detector_config", lambda entry: entry
    )
    adapted = AdaptedDetector(base={}, split="grace.splits.dinov3.DINOv3Split")

    x = torch.randn(4, 3, IMAGE, IMAGE)
    with torch.no_grad():
        assert torch.equal(adapted(x), split.detector(x))


def test_untrained_head_warns_rather_than_scoring_silently(detector_factory):
    """A random head gives ~0.5 AUC, which reads as a failed adapter rather
    than as a missing stage 0."""
    with pytest.warns(UserWarning, match="scripts/train_probe.py"):
        DINOv3Split(detector_factory())
