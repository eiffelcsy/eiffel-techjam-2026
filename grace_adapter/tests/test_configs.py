"""Every shipped config must load into its dataclass.

A typo in a config that silently trains the default objective costs a day and
looks like a negative result, so `_build` raises on unknown keys and this test
walks every file in the repo to prove they all parse.
"""

from pathlib import Path

import pytest
import yaml

from grace.config import (
    load_cache_config, load_discrepancy_config, load_probe_config, load_train_config,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _paths(subdir):
    return sorted(CONFIGS.joinpath(subdir).glob("*.yaml"))


FAMILIES = ["rine", "dinov3"]
"""Detector families with a full config set. Every structural check below runs
over all of them: a PoC config set that quietly diverged from the one the tests
describe would be worse than not having one, since the PoC exists precisely to
be the arm that runs end to end."""


@pytest.mark.parametrize("path", _paths("probe"), ids=lambda p: p.name)
def test_probe_configs_load(path):
    """Stage 0, PoC only. `out: ""` is not a mistake -- it means "take the path
    from the detector config", which is what keeps it written down once."""
    cfg = load_probe_config(path)
    assert cfg.epochs > 0 and cfg.detector and cfg.dataset
    assert cfg.n_layers >= 1


@pytest.mark.parametrize("path", _paths("cache"), ids=lambda p: p.name)
def test_cache_configs_load(path):
    cfg = load_cache_config(path)
    assert cfg.n_epochs > 0 and cfg.split


@pytest.mark.parametrize(
    "path", [p for p in _paths("train") if "discrepancy" not in p.name], ids=lambda p: p.name
)
def test_stage_one_configs_load(path):
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


def test_unknown_key_is_rejected(tmp_path):
    """Silently ignoring a typo is the failure this guards against."""
    path = tmp_path / "typo.yaml"
    path.write_text("run_id: x\ncache_dir: y\ntargetview: clean\n", encoding="utf-8")
    with pytest.raises(KeyError, match="targetview"):
        load_train_config(path)


def test_nested_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(
        "run_id: x\ncache_dir: y\nloss:\n  lam_sliced: 0.1\n", encoding="utf-8"
    )
    with pytest.raises(KeyError, match="lam_sliced"):
        load_train_config(path)


@pytest.mark.parametrize("path", _paths("detectors"), ids=lambda p: p.name)
def test_detector_configs_are_in_the_harness_shape(path):
    """These are read by eval_pipeline, not by grace, so they must match its
    DetectorConfig rather than anything defined here."""
    from pipeline.config import load_detector_config

    cfg = load_detector_config(path)
    assert cfg.target == "grace.detectors.adapted.AdaptedDetector"
    assert "base" in cfg.args and "split" in cfg.args


@pytest.mark.parametrize("family", FAMILIES)
def test_identity_config_has_no_checkpoint(family):
    """E1's whole point: the null adapter must reproduce the baseline exactly."""
    raw = yaml.safe_load((CONFIGS / f"detectors/{family}+identity.yaml").read_text())
    assert raw["args"]["checkpoint"] is None


@pytest.mark.parametrize("family", FAMILIES)
def test_grace_and_grace_d_share_an_adapter(family):
    """Stage 2 never touches the adapter, so the two variants must name the same
    checkpoint -- otherwise the comparison confounds two changes."""
    grace = yaml.safe_load((CONFIGS / f"detectors/{family}+grace.yaml").read_text())
    graced = yaml.safe_load((CONFIGS / f"detectors/{family}+grace-d.yaml").read_text())
    assert grace["args"]["checkpoint"] == graced["args"]["checkpoint"]
    assert grace["args"]["discrepancy"] is None
    assert graced["args"]["discrepancy"] is not None


@pytest.mark.parametrize("family", FAMILIES)
def test_all_three_arms_wrap_the_same_base_detector(family):
    """identity / grace / grace-d must differ in the adapter alone. A different
    `base` or `split` between them makes the comparison meaningless."""
    raws = [
        yaml.safe_load((CONFIGS / f"detectors/{family}+{arm}.yaml").read_text())
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
# Configs reference the harness's own detector and dataset files by relative
# path, which is what stops GRACE from redefining either and drifting from what
# was benchmarked. It also means a rename in eval_pipeline breaks a config here
# silently, at the top of a run rather than at import time.
#
# Paths resolve against the CWD the reading tool runs in, and that differs by
# config kind: grace's own scripts run from grace_adapter/, while run_eval.py
# runs from eval_pipeline/. Both roots are checked below.

HERE = CONFIGS.parent
HARNESS = HERE.parent / "eval_pipeline"


@pytest.mark.parametrize(
    "path", _paths("probe") + _paths("cache") + _paths("train"), ids=lambda p: p.name
)
def test_referenced_detectors_and_datasets_exist(path):
    """Resolved from grace_adapter/, where build_cache.py and train_*.py run."""
    raw = yaml.safe_load(path.read_text())
    for key in ("detector", "dataset", "val_dataset"):
        if raw.get(key):
            assert (HERE / raw[key]).resolve().is_file(), f"{path.name}: {key} -> {raw[key]}"


@pytest.mark.parametrize("path", _paths("detectors"), ids=lambda p: p.name)
def test_adapted_detectors_reference_a_real_base(path):
    """Resolved from eval_pipeline/, where run_eval.py reads these."""
    base = yaml.safe_load(path.read_text())["args"]["base"]
    assert (HARNESS / base).resolve().is_file(), f"{path.name}: base -> {base}"


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


DATASET_CONFIGS = sorted((HARNESS / "configs" / "datasets").glob("*.yaml"))


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_manifest_paths_resolve_the_same_from_either_package(path):
    """A dataset config is read from TWO working directories, and must mean the
    same thing in both.

    `run_eval.py` runs with the CWD at `eval_pipeline/`; `build_cache.py`,
    `train_probe.py` and `train_adapter.py` run with the CWD at `grace_adapter/`.
    Config paths resolve against the CWD, so a bare `data/sid_poc/...` names
    `eval_pipeline/data/...` for one caller and `grace_adapter/data/...` for the
    other -- it works for whoever built it and raises `FileNotFoundError` for
    everyone else. That is exactly the shape of bug this catches.

    `data/` therefore lives at the repo root and configs say `../data/...`, which
    is the same directory from either sibling. Existence is deliberately NOT
    asserted: manifests are gitignored, materialized by `build_manifest.py`, and
    absent on a fresh clone. Agreement is the invariant, not presence.
    """
    manifest = yaml.safe_load(path.read_text())["manifest"]
    from_harness = (HARNESS / manifest).resolve()
    from_grace = (HERE / manifest).resolve()
    assert from_harness == from_grace, (
        f"{path.name}: manifest {manifest!r} resolves to\n"
        f"  {from_harness}   (CWD=eval_pipeline, run_eval.py)\n"
        f"  {from_grace}   (CWD=grace_adapter, train_*.py / build_cache.py)\n"
        f"Use a repo-root-relative '../data/...' so both agree."
    )
    assert from_harness.parent.parent == HERE.parent / "data", (
        f"{path.name}: manifests belong under the repo-root data/ directory, "
        f"got {from_harness}"
    )
