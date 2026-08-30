"""Every shipped config must load into its dataclass.

A typo in a config that silently trains the default objective costs a day and
looks like a negative result, so `_build` raises on unknown keys and this test
walks every file in the repo to prove they all parse.
"""

from pathlib import Path

import pytest
import yaml

from train.config import (
    load_cache_config, load_discrepancy_config, load_enrich_config, load_probe_config,
    load_train_config,
)

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS = ROOT / "train" / "configs"          # probe/, cache/, train/, defaults.yaml
DETECTORS = ROOT / "eval" / "configs" / "detectors"


def _paths(subdir):
    root = DETECTORS if subdir == "detectors" else CONFIGS.joinpath(subdir)
    return sorted(root.glob("*.yaml"))


def _adapted_detector_paths():
    """This package's own detector configs within eval/configs/detectors/ --
    the ones naming an AdaptedDetector/FusedDetector, not the plain harness
    detectors (dinov3-wildfake*.yaml) that merged into the same directory.
    Named `<base>+<arm>.yaml` throughout the project, so a literal `+` is the
    discriminator."""
    return sorted(p for p in DETECTORS.glob("*.yaml") if "+" in p.name)


FAMILIES = ["dinov3"]
"""Detector families with a full config set. Every structural check below runs
over all of them: a PoC config set that quietly diverged from the one the tests
describe would be worse than not having one, since the PoC exists precisely to
be the arm that runs end to end.

ONE family now. `rine` was the second, and it was the only evidence that the
config structure -- and the method behind it -- is not specific to a single
seam. It went with the detector zoo. These checks still catch a config set that
drifts internally; they can no longer catch one that is shaped around DINOv3
alone, because there is nothing left to compare against."""


CROP_CONFIGS = {
    "probe/dinov3_wildfake_multiscale.yaml",
    "cache/dinov3_multiscale.yaml",
    "cache/dinov3_multiscale_val.yaml",
    "cache/wildfake_freq.yaml",
    "cache/wildfake_freq_val.yaml",
    "train/dinov3_multiscale.yaml",
    "train/dinov3_enrich.yaml",
}
"""Every config carrying the multi-scale window protocol.

Seven files rather than one because the number has to be restated wherever the
protocol is: stage 0 fits the head on those windows, four caches render them, and
two training runs fingerprint them. `after_fetch.sh --write-range` writes all
seven from one audit; `test_every_crop_config_agrees_on_the_range` is what stops
them drifting apart afterwards."""

AWAITING_AUDIT: set[str] = set()
"""The subset of CROP_CONFIGS that must still REFUSE to load.

`crop.s_max` is the largest window every image in the corpus can supply from its
own pixels, and it is a property of the corpus rather than of the method: draw
beyond it and one class clamps more than the other, so the realized crop size
becomes a classifier and the augmentation hands back the shortcut it was added to
remove. On wildfake_test an unaudited 128-512 range scores 0.9895 that way. So
these ship with `s_max` absent and raise on load -- the refusal IS the feature.

EMPTIED 2026-08-30, when `audit_sizes.py` measured the training corpus and
`--write-range` wrote s_max: 256 into all seven. It is deliberately still here
and still separate from CROP_CONFIGS: the next corpus needs the same forcing
function, and re-populating this set is how it gets one."""

STAGE_TWO = {"discrepancy", "enrich"}
"""Filename markers for the two stage-2 families, which load through their own
dataclasses. `train/` holds three kinds of run now, not two."""


def _audited(subdir, path) -> bool:
    return f"{subdir}/{path.name}" in AWAITING_AUDIT


@pytest.mark.parametrize("path", _paths("probe"), ids=lambda p: p.name)
def test_probe_configs_load(path):
    """Stage 0, PoC only. `out: ""` is not a mistake -- it means "take the path
    from the detector config", which is what keeps it written down once."""
    if _audited("probe", path):
        pytest.skip("refuses to load until audited; see test_the_crop_range_must_be_audited")
    cfg = load_probe_config(path)
    assert cfg.epochs > 0 and cfg.detector and cfg.dataset
    assert cfg.n_layers >= 1


LOADERS = {
    "probe": load_probe_config,
    "cache": load_cache_config,
    "train": load_train_config,
}


@pytest.mark.parametrize("name", sorted(AWAITING_AUDIT))
def test_the_crop_range_must_be_audited(name):
    """The forcing function, tested rather than trusted.

    Once `scripts/after_fetch.sh --write-range` writes the audited `s_max` in,
    this test starts failing -- and that failure is the signal to empty
    AWAITING_AUDIT, not to soften the check."""
    subdir, filename = name.split("/")
    loader = LOADERS[subdir]
    if subdir == "train" and any(m in filename for m in STAGE_TWO):
        loader = load_enrich_config if "enrich" in filename else load_discrepancy_config
    with pytest.raises(ValueError, match="audit_sizes"):
        loader(CONFIGS / subdir / filename)


def test_every_crop_config_agrees_on_the_range():
    """One measurement, restated in seven files, and they must not drift.

    The crop range is a property of the CORPUS, so a stage-0 head fit at 128-448
    and a cache rendered at 128-512 are not two settings of a knob -- they are a
    head scored on a feature space it was never fit on, which is exactly what
    `crop_sha` and `_assert_head_matches` exist to refuse one layer down. Here it
    is caught before anything runs.

    Reads the YAML rather than the dataclass so it still says something while
    `s_max` is unset: seven files with no `s_max` agree, vacuously and correctly.

    Ranges over CROP_CONFIGS, not AWAITING_AUDIT. Those were one set until the
    audit landed, at which point iterating the empty one would have quietly
    turned this into a test of nothing -- exactly when it began to matter.
    """
    ranges = {}
    for name in sorted(CROP_CONFIGS):
        raw = yaml.safe_load((CONFIGS / name).read_text()).get("crop") or {}
        ranges[name] = (raw.get("s_min"), raw.get("s_max"), raw.get("policy"),
                        raw.get("seed"))
    assert len(set(ranges.values())) == 1, (
        f"the crop range differs across configs that must share it:\n  "
        + "\n  ".join(f"{k}: s_min={v[0]} s_max={v[1]} policy={v[2]} seed={v[3]}"
                      for k, v in ranges.items())
    )


@pytest.mark.parametrize("path", _paths("cache"), ids=lambda p: p.name)
def test_cache_configs_load(path):
    if _audited("cache", path):
        pytest.skip("refuses to load until audited; see test_the_crop_range_must_be_audited")
    cfg = load_cache_config(path)
    assert cfg.n_epochs > 0 and cfg.split


@pytest.mark.parametrize(
    "path",
    [p for p in _paths("train") if not any(m in p.name for m in STAGE_TWO)],
    ids=lambda p: p.name,
)
def test_stage_one_configs_load(path):
    if _audited("train", path):
        pytest.skip("refuses to load until audited; see test_the_crop_range_must_be_audited")
    cfg = load_train_config(path)
    assert cfg.target_view in ("clean", "degraded")
    assert cfg.source in ("cache", "live")
    assert cfg.loss.weighting in ("none", "jacobian")
    assert 0.0 <= cfg.loss.eps_iso <= 1.0


@pytest.mark.parametrize(
    "path", [p for p in _paths("train") if "discrepancy" in p.name], ids=lambda p: p.name
)
def test_stage_two_configs_load(path):
    assert load_discrepancy_config(path).adapter_checkpoint


@pytest.mark.parametrize(
    "path", [p for p in _paths("train") if "enrich" in p.name], ids=lambda p: p.name
)
def test_enrich_configs_load(path):
    if _audited("train", path):
        pytest.skip("refuses to load until audited; see test_the_crop_range_must_be_audited")
    cfg = load_enrich_config(path)
    assert cfg.adapter_checkpoint and cfg.freq.enabled
    assert cfg.lam_anchor >= 0


def test_an_enrich_config_without_the_freq_view_is_rejected(tmp_path):
    """The whole stage reads the frequency view, so `freq.enabled: false` is not
    a smaller run -- it is a run whose every step would fail on a missing memmap
    several minutes in, or worse, train an enricher over zeros."""
    path = tmp_path / "no_freq.yaml"
    path.write_text(
        "run_id: x\ncache_dir: y\nadapter_checkpoint: z\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="freq.enabled"):
        load_enrich_config(path)


def test_unknown_key_is_rejected(tmp_path):
    """Silently ignoring a typo is the failure this guards against."""
    path = tmp_path / "typo.yaml"
    path.write_text("run_id: x\ncache_dir: y\ntargetview: clean\n", encoding="utf-8")
    with pytest.raises(KeyError, match="targetview"):
        load_train_config(path)


def test_nested_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(
        "run_id: x\ncache_dir: y\nloss:\n  lam_kel: 0.1\n", encoding="utf-8"
    )
    with pytest.raises(KeyError, match="lam_kel"):
        load_train_config(path)


GRACE_DETECTORS = {
    "grace_adapter.detectors.adapted.AdaptedDetector",
    "freq_branch.detectors.fused.FusedDetector",
}
"""The two detector classes this package puts into the harness.

`AdaptedDetector` takes one tensor and walks the harness's original path.
`FusedDetector` takes an `Inputs` pair, because the DCT branch has to read the
image at native pixel scale and that does not survive preprocessing. Anything
else appearing here is a config pointed at a class the harness cannot build."""


@pytest.mark.parametrize("path", _adapted_detector_paths(), ids=lambda p: p.name)
def test_detector_configs_are_in_the_harness_shape(path):
    """These are read by eval_pipeline, not by grace, so they must match its
    DetectorConfig rather than anything defined here."""
    from eval.config import load_detector_config

    cfg = load_detector_config(path)
    assert cfg.target in GRACE_DETECTORS
    assert "base" in cfg.args and "split" in cfg.args


@pytest.mark.parametrize(
    "arm", ["crop200", "r512"]
)
def test_freq_arms_differ_from_their_control_in_one_key(arm):
    """E10 and E12 are read off three files that must be one key apart.

    `+grace` is the control, `+grace-freq-null` must reproduce it exactly, and
    `+grace-freq` is the measurement. If they wrapped different bases or loaded
    different adapters, the difference between their result files would be two
    changes and neither experiment would mean anything.
    """
    raws = {
        suffix: yaml.safe_load(
            (DETECTORS / f"dinov3-{arm}+{suffix}.yaml").read_text()
        )
        for suffix in ("grace", "grace-freq-null", "grace-freq")
    }
    assert len({r["args"]["base"] for r in raws.values()}) == 1
    assert len({r["args"]["split"] for r in raws.values()}) == 1
    assert len({r["args"]["checkpoint"] for r in raws.values()}) == 1
    assert raws["grace-freq-null"]["args"]["enricher"] is None
    assert raws["grace-freq"]["args"]["enricher"] is not None
    # The base must be the eval arm this file is named for, or the two-column
    # table silently reports one arm twice.
    assert raws["grace"]["args"]["base"].endswith(f"dinov3-wildfake-{arm}.yaml")


@pytest.mark.parametrize("family", FAMILIES)
def test_identity_config_has_no_checkpoint(family):
    """E1's whole point: the null adapter must reproduce the baseline exactly."""
    raw = yaml.safe_load((DETECTORS / f"{family}+identity.yaml").read_text())
    assert raw["args"]["checkpoint"] is None


@pytest.mark.parametrize("family", FAMILIES)
def test_grace_and_grace_d_share_an_adapter(family):
    """Stage 2 never touches the adapter, so the two variants must name the same
    checkpoint -- otherwise the comparison confounds two changes."""
    grace = yaml.safe_load((DETECTORS / f"{family}+grace.yaml").read_text())
    graced = yaml.safe_load((DETECTORS / f"{family}+grace-d.yaml").read_text())
    assert grace["args"]["checkpoint"] == graced["args"]["checkpoint"]
    assert grace["args"]["discrepancy"] is None
    assert graced["args"]["discrepancy"] is not None


@pytest.mark.parametrize("family", FAMILIES)
def test_all_three_arms_wrap_the_same_base_detector(family):
    """identity / grace / grace-d must differ in the adapter alone. A different
    `base` or `split` between them makes the comparison meaningless."""
    raws = [
        yaml.safe_load((DETECTORS / f"{family}+{arm}.yaml").read_text())
        for arm in ("identity", "grace", "grace-d")
    ]
    assert len({r["args"]["base"] for r in raws}) == 1
    assert len({r["args"]["split"] for r in raws}) == 1


@pytest.mark.parametrize("family", FAMILIES)
def test_arms_differ_only_in_target_view(family):
    """Arm A and arm B must be one key apart, or the ablation is not an ablation."""
    a = yaml.safe_load((CONFIGS / f"train/{family}_degraded.yaml").read_text())
    b = yaml.safe_load((CONFIGS / f"train/{family}_clean.yaml").read_text())
    ignored = {"run_id", "target_view", "checkpoint_every", "loss"}
    assert {k: v for k, v in a.items() if k not in ignored} == {
        k: v for k, v in b.items() if k not in ignored
    }
    assert (a["target_view"], b["target_view"]) == ("degraded", "clean")


def test_defaults_yaml_is_documentation_only():
    """It concatenates every config kind, so it must never be loadable as one."""
    raw = yaml.safe_load((CONFIGS / "defaults.yaml").read_text())
    assert "run_id" in raw and "n_epochs" in raw and "adapter_checkpoint" in raw


# ----------------------------------------------------- cross-file references --
# Configs reference the eval stage's own detector and dataset files by relative
# path, which is what stops GRACE from redefining either and drifting from what
# was benchmarked. It also means a rename under load_data/ or eval/ breaks a
# config here silently, at the top of a run rather than at import time.
#
# Every script in the project is now run from the repo root, so every path
# resolves from there -- unlike the two-CWD split this test used to check.



@pytest.mark.parametrize(
    "path", _paths("probe") + _paths("cache") + _paths("train"), ids=lambda p: p.name
)
def test_referenced_detectors_and_datasets_exist(path):
    raw = yaml.safe_load(path.read_text())
    for key in ("detector", "dataset", "val_dataset"):
        value = raw.get(key)
        if not value:
            continue
        # `val_dataset` takes one path or several. One today: the held-out
        # `validation` split of the training manifest.
        for ref in [value] if isinstance(value, str) else value:
            assert (ROOT / ref).resolve().is_file(), f"{path.name}: {key} -> {ref}"
    for key in ("val_datasets", "val_cache_dirs"):
        for ref in raw.get(key) or []:
            if key == "val_datasets":
                assert (ROOT / ref).resolve().is_file(), f"{path.name}: {key} -> {ref}"


@pytest.mark.parametrize("path", _adapted_detector_paths(), ids=lambda p: p.name)
def test_adapted_detectors_reference_a_real_base(path):
    base = yaml.safe_load(path.read_text())["args"]["base"]
    assert (ROOT / base).resolve().is_file(), f"{path.name}: base -> {base}"


@pytest.mark.parametrize(
    "loader,body",
    [
        (load_train_config, "run_id: x\ncache_dir: y\n"),
        (load_discrepancy_config, "run_id: x\ncache_dir: y\nadapter_checkpoint: z\n"),
    ],
    ids=["stage1", "stage2"],
)
def test_log_every_zero_is_rejected_at_load(tmp_path, loader, body):
    """`step % 0` would be a ZeroDivisionError several minutes into a run, and 0
    is the natural typo for "never log" -- which is not on offer, because the
    history in summary.json is written from these same points."""
    path = tmp_path / "bad.yaml"
    path.write_text(body + "log_every: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="log_every"):
        loader(path)


def test_negative_val_every_is_rejected_at_load(tmp_path):
    """0 is the documented "validate once, at the end". A negative would just
    never fire, silently producing a run with no mid-training curve at all."""
    path = tmp_path / "bad.yaml"
    path.write_text("run_id: x\ncache_dir: y\nval_every: -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="val_every"):
        load_train_config(path)


DATASET_CONFIGS = sorted((ROOT / "load_data" / "configs" / "datasets").glob("*.yaml"))


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_manifest_paths_resolve_under_repo_root_data(path):
    """Every script in the project runs with the CWD at the repo root now, so a
    dataset config's `manifest:` must resolve from there and land under the
    shared `data/` directory -- not a package-relative path that only worked
    for whichever script happened to `cd` into its own package first.

    Existence is deliberately NOT asserted: manifests are gitignored,
    materialized by `build_manifest.py`, and absent on a fresh clone.
    """
    manifest = yaml.safe_load(path.read_text())["manifest"]
    resolved = (ROOT / manifest).resolve()
    assert resolved.parent.parent == ROOT / "data", (
        f"{path.name}: manifests belong under the repo-root data/ directory, "
        f"got {resolved}"
    )


@pytest.mark.parametrize("name", sorted(CROP_CONFIGS))
def test_crop_configs_load_once_audited(name):
    """The other side of the forcing function.

    While a config is in AWAITING_AUDIT it must refuse; once the audit has
    written `s_max` it must load AND carry a real range. Without this, emptying
    AWAITING_AUDIT would leave nothing asserting these files work at all.
    """
    if name in AWAITING_AUDIT:
        pytest.skip("still awaiting the audit")
    subdir, filename = name.split("/")
    loader = LOADERS[subdir]
    if subdir == "train" and any(m in filename for m in STAGE_TWO):
        loader = load_enrich_config if "enrich" in filename else load_discrepancy_config
    cfg = loader(CONFIGS / subdir / filename)
    assert cfg.crop.enabled
    assert cfg.crop.s_max is not None and cfg.crop.s_max >= cfg.crop.s_min
