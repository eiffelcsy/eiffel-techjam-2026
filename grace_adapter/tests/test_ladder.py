"""The ladder, end to end: split -> cache -> loader -> adapter -> checkpoint.

The ladder's failure modes are almost all silent. A tap read from the wrong view
still trains. A tap crossed with another row still trains. A ladder checkpoint
scored against a split tapping different blocks still produces logits. None of
them crash, and each produces a plausible retention curve for a model that is not
the one being claimed -- so the checks that matter here are the alignment ones,
not the shape ones.

`tests/test_dinov3_taps.py` covers what only a real ViT can: that the tapped
forward reproduces the seam exactly. This file covers everything downstream of
that, on the toy split, and runs with no weights.
"""

import numpy as np
import pytest
import torch

from grace.cache.reader import FeatureCache
from grace.cache.schedule import EpochSchedule, val_epochs
from grace.cache.spec import (
    CacheSpec, sha_manifest, sha_preprocess, tap_view_name, view_name,
)
from grace.cache.writer import build_cache, is_complete
from grace.config import AdapterConfig, LossConfig, TrainConfig
from grace.models.adapter import GATE_INIT, GatedResidualAdapter
from grace.models.factory import build_adapter, load_adapter, save_adapter
from grace.models.ladder import LadderAdapter, tap_spec_for
from grace.splits.base import FeatureSpec
from grace.splits.verify import verify_taps
from grace.train.data import build_loader
from grace.train.loop import train_adapter
from preprocessing.degrade.conditions import load_grid
from tests.fixtures import SPECS, MLPHead, ToySplit, features, write_images

N_IMAGES = 24
N_TAPS = 3
GRID_FILE = "preprocessing/configs/degradations.yaml"


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """A real cache carrying tap views, over generated images."""
    root = tmp_path_factory.mktemp("ladder")
    manifest = write_images(root / "images", N_IMAGES)
    spec = SPECS["vector"]
    split = ToySplit(spec, head=MLPHead(spec), n_taps=N_TAPS, verify=True)
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    epochs = [0, 1, *val_epochs(1)]

    cache_spec = CacheSpec(
        detector="toy",
        feature=spec,
        n=len(manifest),
        shard_size=10,                       # forces a multi-shard render
        manifest_sha=sha_manifest(manifest),
        schedule_sha=schedule.fingerprint(),
        detector_sha="toy",
        preprocess_sha=sha_preprocess(split.preprocess_fn()),
        taps=split.taps(),
        tap_feature=split.tap_spec(),
    )
    out = root / "toy"
    build_cache(split, manifest, out, cache_spec, schedule, epochs,
                batch_size=4, num_workers=0)
    return root, out, manifest, split, schedule


# --------------------------------------------------------------------------
# the module


def test_ladder_is_identity_at_init_whatever_the_taps():
    """The guarantee the whole project rests on, with the tap pathway live.

    If this fails, a clean-AUC change is unattributable: the ladder might have
    corrected badly, or merely perturbed.
    """
    spec, tspec = SPECS["vector"], FeatureSpec(layout="layers", shape=(N_TAPS, 16))
    lad = build_adapter(spec, AdapterConfig(taps=True), tspec, ("a", "b", "c"))
    f = features(spec, batch=6)
    for taps in (None, torch.randn(6, N_TAPS, 16), torch.randn(6, N_TAPS, 16) * 1e3):
        assert torch.equal(lad(f, taps=taps), f)
        assert torch.equal(lad(f, taps=taps, severity=torch.rand(6)), f)


def test_taps_actually_change_the_correction():
    """Identity at init must not be the ladder doing nothing forever.

    Zero-init `fc2` makes every correction zero, so this perturbs it first --
    otherwise the test would pass on an adapter whose tap pathway was
    disconnected entirely.
    """
    spec, tspec = SPECS["vector"], FeatureSpec(layout="layers", shape=(N_TAPS, 16))
    lad = build_adapter(spec, AdapterConfig(taps=True), tspec)
    for layer in lad.fc2:
        torch.nn.init.normal_(layer.weight, std=0.05)

    f = features(spec, batch=6)
    a = lad(f, taps=torch.randn(6, N_TAPS, 16))
    b = lad(f, taps=torch.randn(6, N_TAPS, 16))
    assert not torch.allclose(a, b), "the correction ignores its taps"
    assert not torch.allclose(a, lad(f)), "taps=None is not distinguishable"


def test_every_ladder_parameter_gets_gradient_once_training_starts():
    """A hard-zero tap gate would leave `tap_proj` permanently starved.

    `fc2` is zero-initialized, so at step 0 nothing upstream of it receives
    gradient -- that is the base adapter's design, and `fc2` moving off zero is
    what starts everything else. So this perturbs `fc2` first and asks the
    question that actually matters: once the adapter is learning at all, does
    gradient reach the whole tap pathway? A `tap_gate` initialized to zero rather
    than to `GATE_INIT` would put a second multiplicative zero in the way and
    fail here.
    """
    spec, tspec = SPECS["vector"], FeatureSpec(layout="layers", shape=(N_TAPS, 16))
    lad = build_adapter(spec, AdapterConfig(taps=True, severity_film=False), tspec)
    for layer in lad.fc2:
        torch.nn.init.normal_(layer.weight, std=0.05)

    lad(features(spec, batch=6), taps=torch.randn(6, N_TAPS, 16)).pow(2).mean().backward()

    starved = [
        name for name, p in lad.named_parameters()
        if p.grad is None or p.grad.abs().max() == 0
    ]
    assert not starved, f"no gradient reaches {starved}"


def test_plain_adapter_refuses_taps():
    """Silently ignoring them would burn a rendered tap cache and report an
    honest-looking number for the wrong model."""
    plain = build_adapter(SPECS["vector"], AdapterConfig())
    assert not plain.reads_taps
    with pytest.raises(ValueError, match="no ladder"):
        plain(features(SPECS["vector"], batch=4), taps=torch.randn(4, N_TAPS, 16))


def test_ladder_refuses_the_wrong_tap_count():
    spec, tspec = SPECS["vector"], FeatureSpec(layout="layers", shape=(N_TAPS, 16))
    lad = build_adapter(spec, AdapterConfig(taps=True), tspec)
    with pytest.raises(ValueError, match="expected taps of shape"):
        lad(features(spec, batch=4), taps=torch.randn(4, N_TAPS + 1, 16))


def test_tap_spec_for_refuses_a_split_with_no_taps():
    with pytest.raises(ValueError, match="emits no taps"):
        tap_spec_for(ToySplit(SPECS["vector"]), AdapterConfig(taps=True))
    assert tap_spec_for(ToySplit(SPECS["vector"]), AdapterConfig(taps=False)) is None


def test_verify_taps_catches_a_split_that_does_not_reproduce_its_seam():
    """The corrupting failure: a tapped forward that returns different features.

    Every downstream number would be a comparison against a model no baseline
    was ever measured at, and nothing would crash.
    """
    class Liar(ToySplit):
        def trunk_with_taps(self, x):
            f, taps = super().trunk_with_taps(x)
            return f + 1.0, taps

    with pytest.raises(RuntimeError, match="does not reproduce trunk"):
        verify_taps(Liar(SPECS["vector"], n_taps=N_TAPS))


# --------------------------------------------------------------------------
# checkpoints


def test_checkpoint_round_trips_as_a_ladder(tmp_path):
    """A checkpoint must rebuild into the class it was saved from with no
    reference to the run that produced it -- that is what lets a detector config
    name a checkpoint and nothing else."""
    spec, tspec = SPECS["vector"], FeatureSpec(layout="layers", shape=(N_TAPS, 16))
    names = ("block00", "block02", "block04")
    lad = build_adapter(spec, AdapterConfig(taps=True), tspec, names)
    for layer in lad.fc2:
        torch.nn.init.normal_(layer.weight, std=0.05)

    path = tmp_path / "ema.pt"
    save_adapter(path, lad, spec, AdapterConfig(taps=True))
    back = load_adapter(str(path), spec, tspec)

    assert isinstance(back, LadderAdapter)
    assert back.tap_names == names
    f, taps = features(spec, batch=5), torch.randn(5, N_TAPS, 16)
    assert torch.equal(back(f, taps=taps), lad(f, taps=taps))


def test_loading_refuses_a_ladder_against_a_mismatched_tap_set(tmp_path):
    spec = SPECS["vector"]
    lad = build_adapter(
        spec, AdapterConfig(taps=True), FeatureSpec(layout="layers", shape=(N_TAPS, 16))
    )
    path = tmp_path / "ema.pt"
    save_adapter(path, lad, spec, AdapterConfig(taps=True))

    with pytest.raises(ValueError, match="tap sets must match"):
        load_adapter(str(path), spec, FeatureSpec(layout="layers", shape=(N_TAPS + 2, 16)))


def test_loading_refuses_a_plain_checkpoint_against_a_tapping_split(tmp_path):
    """Otherwise the ladder arm would quietly score the plain adapter."""
    spec = SPECS["vector"]
    plain = build_adapter(spec, AdapterConfig())
    path = tmp_path / "plain.pt"
    save_adapter(path, plain, spec, AdapterConfig())

    assert isinstance(load_adapter(str(path), spec), GatedResidualAdapter)
    with pytest.raises(ValueError, match="plain adapter but this split emits taps"):
        load_adapter(str(path), spec, FeatureSpec(layout="layers", shape=(N_TAPS, 16)))


# --------------------------------------------------------------------------
# the cache


def test_tap_views_are_rendered_and_finalized(rendered):
    _, out, _, _, _ = rendered
    for epoch in (None, 0, 1, *val_epochs(1)):
        assert is_complete(out / tap_view_name(epoch)), f"taps for {view_name(epoch)}"
    assert CacheSpec.load(out).taps == ("block00", "block01", "block02")


def test_tap_directories_do_not_pollute_the_epoch_glob(rendered):
    """`FeatureCache.epochs()` parses `epoch=*` with `int()`. A sibling tap
    naming would be picked up by that glob and crash -- or worse, be counted."""
    _, out, _, _, _ = rendered
    assert FeatureCache(out).epochs() == (0, 1, *val_epochs(1))


def test_cached_taps_match_a_live_forward_row_for_row(rendered):
    """The alignment check, which is the whole risk: tap row `r` must be image
    `r`'s taps, from the same view its features came from."""
    _, out, manifest, split, schedule = rendered
    cache = FeatureCache(out)
    rng = np.random.default_rng(0)
    picked = rng.choice(np.asarray(manifest.index), size=8, replace=False)

    for epoch in (None, 1):
        cached = cache.clean_taps(picked) if epoch is None else cache.taps(picked, epoch)
        for row, idx in enumerate(picked):
            img = _image(manifest, idx)
            if epoch is not None:
                img, _ = schedule.apply(img, int(idx), epoch)
            with torch.no_grad():
                _, live = split.trunk_with_taps(
                    split.preprocess_fn()(img).unsqueeze(0)
                )
            assert torch.allclose(cached[row].float(), live[0], atol=1e-2)


def test_reading_taps_from_a_tapless_cache_is_an_error(tmp_path):
    manifest = write_images(tmp_path / "images", 8)
    spec = SPECS["vector"]
    split = ToySplit(spec)
    schedule = EpochSchedule(grid=load_grid(GRID_FILE), seed=0)
    cache_spec = CacheSpec(detector="toy", feature=spec, n=len(manifest), shard_size=8)
    out = tmp_path / "plain"
    build_cache(split, manifest, out, cache_spec, schedule, [0],
                batch_size=4, num_workers=0)

    cache = FeatureCache(out)
    assert not cache.has_taps
    with pytest.raises(FileNotFoundError, match="rendered without taps"):
        cache.taps(np.asarray(manifest.index)[:2], 0)


def test_a_subset_of_the_cached_taps_can_be_read_without_re_rendering(rendered):
    """Dropping a tap is a column selection, not a re-render.

    This is what makes "block00 turned out to be inert" a config change rather
    than another pass over 277k images: the bytes are already on disk, and the
    run simply stops reading that column.
    """
    _, out, _, split, _ = rendered
    want = ("block00", "block02")                      # drop the middle tap
    expect = CacheSpec(
        detector="toy", feature=SPECS["vector"], n=0,
        taps=want, tap_feature=FeatureSpec(layout="layers", shape=(2, 16)),
    )
    full = FeatureCache(out)
    sub = FeatureCache(out, expect=expect)

    assert full.taps_selected == ("block00", "block01", "block02")
    assert sub.taps_selected == want
    assert sub.tap_feature.shape == (2, 16)
    assert full.tap_feature.shape == (3, 16)

    idx = np.asarray(full.index)[:6]
    a, b = full.taps(idx, 0), sub.taps(idx, 0)
    assert b.shape == (6, 2, 16)
    # The columns must be the ORIGINAL ones, not the first two.
    assert torch.equal(b[:, 0], a[:, 0])
    assert torch.equal(b[:, 1], a[:, 2])


def test_subset_selection_follows_the_requested_order(rendered):
    """The adapter's per-tap gates are indexed by position, so a reader that
    returned cache order rather than requested order would attach every gate to
    the wrong block -- silently, and with plausible numbers."""
    _, out, _, _, _ = rendered
    want = ("block02", "block00")                      # deliberately reversed
    expect = CacheSpec(
        detector="toy", feature=SPECS["vector"], n=0,
        taps=want, tap_feature=FeatureSpec(layout="layers", shape=(2, 16)),
    )
    full = FeatureCache(out)
    sub = FeatureCache(out, expect=expect)
    assert sub.taps_selected == want

    idx = np.asarray(full.index)[:4]
    a, b = full.taps(idx, 0), sub.taps(idx, 0)
    assert torch.equal(b[:, 0], a[:, 2])
    assert torch.equal(b[:, 1], a[:, 0])


def test_requesting_every_cached_tap_is_a_no_op(rendered):
    _, out, _, split, _ = rendered
    expect = CacheSpec(
        detector="toy", feature=SPECS["vector"], n=0,
        taps=split.taps(), tap_feature=split.tap_spec(),
    )
    sub = FeatureCache(out, expect=expect)
    assert sub._tap_select is None, "identity selection should not index at all"
    assert sub.tap_feature.shape == split.tap_spec().shape


def test_a_tap_the_cache_never_rendered_is_still_refused(rendered):
    """Subset semantics must not become 'anything goes'. Dropping a tap is
    free; adding one needs the bytes, and they are not there."""
    _, out, _, _, _ = rendered
    expect = CacheSpec(
        detector="toy", feature=SPECS["vector"], n=0,
        taps=("block00", "block09"),                   # block09 was never rendered
        tap_feature=FeatureSpec(layout="layers", shape=(2, 16)),
    )
    with pytest.raises(ValueError, match="never rendered"):
        FeatureCache(out, expect=expect)


def test_a_tap_of_the_wrong_width_is_refused(rendered):
    """Selecting a subset changes how MANY taps are read, never how wide."""
    _, out, _, _, _ = rendered
    expect = CacheSpec(
        detector="toy", feature=SPECS["vector"], n=0,
        taps=("block00",), tap_feature=FeatureSpec(layout="layers", shape=(1, 32)),
    )
    with pytest.raises(ValueError, match="tap width mismatch"):
        FeatureCache(out, expect=expect)


def test_a_ladder_run_refuses_a_cache_rendered_for_other_blocks(rendered):
    """The mismatch that would otherwise train happily on the wrong trunk depth."""
    _, out, _, _, _ = rendered
    other = CacheSpec(
        detector="toy", feature=SPECS["vector"], n=0,
        taps=("block00", "block09"),
        tap_feature=FeatureSpec(layout="layers", shape=(2, 16)),
    )
    with pytest.raises(ValueError, match="tap mismatch"):
        FeatureCache(out, expect=other)


def test_loader_serves_the_degraded_tap_view(rendered):
    """Only the degraded taps: nothing in the objective reads the clean ones, so
    loading them would be a memmap read per batch that no term consumes."""
    _, out, manifest, _, _ = rendered
    cfg = TrainConfig(run_id="t", cache_dir=str(out), batch_size=4, num_workers=0)
    batch = next(iter(build_loader(cfg, FeatureCache(out), manifest, None, 0,
                                   with_taps=True)))
    assert batch["taps_deg"].shape == (4, N_TAPS, 16)
    assert "taps_clean" not in batch


def test_the_cache_still_renders_the_clean_tap_view(rendered):
    """Rendered even though training does not read it: analysis scripts want the
    undamaged taps, and re-rendering a cache to get them is the expensive path."""
    _, out, manifest, _, _ = rendered
    cache = FeatureCache(out)
    idx = manifest.index.to_numpy()[:4]
    clean, deg = cache.clean_taps(idx), cache.taps(idx, 0)
    assert clean.shape == deg.shape == (4, N_TAPS, 16)
    assert not torch.allclose(clean.float(), deg.float())


# --------------------------------------------------------------------------
# the loop


def test_stage_one_trains_a_ladder_and_moves_its_tap_gates(rendered):
    """The smoke test: a real cache, a real loop, a real checkpoint.

    Asserting the tap gates *moved* is what distinguishes a wired ladder from
    one whose taps are read, projected and then multiplied by a gate that never
    receives gradient.
    """
    root, out, manifest, split, schedule = rendered
    cfg = TrainConfig(
        run_id="ladder", cache_dir=str(out), epochs=1, batch_size=8,
        warmup_steps=1, num_workers=0, log_every=1,
        out_dir=str(root / "checkpoints"),
        adapter=AdapterConfig(taps=True, tap_dim=8, bottleneck=16),
        loss=LossConfig(lam_kl=0.0),
    )
    summary = train_adapter(cfg, split, manifest, schedule)

    gates = {k: v for k, v in summary["history"][-1].items() if k.startswith("tap_gate/")}
    assert set(gates) == {f"tap_gate/block{k:02d}" for k in range(N_TAPS)}

    # The taps start at one shared GATE_INIT, so after any real optimizer step
    # they must have (a) moved and (b) moved by DIFFERENT amounts. The second is
    # the load-bearing half: a ladder whose taps were summed before the gate, or
    # whose gate were broadcast rather than per-tap, would move them identically.
    # A three-step smoke run moves them by ~1e-6, so this checks separation
    # rather than magnitude -- convergence is not what is being tested.
    init = torch.sigmoid(torch.tensor(GATE_INIT)).item()
    assert all(v != init for v in gates.values()), f"tap gates never moved: {gates}"
    assert len(set(gates.values())) == N_TAPS, f"tap gates move as one: {gates}"

    reloaded = load_adapter(
        str(root / "checkpoints" / "ladder" / "ema.pt"), split.feature_spec, split.tap_spec()
    )
    assert isinstance(reloaded, LadderAdapter)


def _image(manifest, idx):
    from preprocessing.dataset import load_normalized

    return load_normalized(manifest.loc[idx, "path"])


def test_gate_init_reaches_the_tap_gate_too():
    """One config key, both gates. `tap_gate` is the one the non-zero init exists
    for -- it multiplies `tap_proj`, so a ladder arm sweeping `gate_init` and
    leaving `tap_gate` at the default would be testing the wrong parameter."""
    spec, tspec = SPECS["vector"], FeatureSpec(layout="layers", shape=(N_TAPS, 16))
    lad = build_adapter(spec, AdapterConfig(taps=True, gate_init=-3.0), tspec)

    want = torch.sigmoid(torch.tensor(-3.0))
    assert torch.allclose(lad.gate(), want, atol=1e-6)
    assert torch.allclose(lad.tap_gate(), want, atol=1e-6)
