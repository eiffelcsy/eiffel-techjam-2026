"""Every shipped config must load into its dataclass.

A typo in a config that silently trains the default objective costs a day and
looks like a negative result, so `_build` raises on unknown keys and this test
walks every file in the repo to prove they all parse.
"""

from pathlib import Path

import pytest
import yaml

from grace.config import load_cache_config, load_discrepancy_config, load_train_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _paths(subdir):
    return sorted(CONFIGS.joinpath(subdir).glob("*.yaml"))


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


def test_identity_config_has_no_checkpoint():
    """E1's whole point: the null adapter must reproduce the baseline exactly."""
    raw = yaml.safe_load((CONFIGS / "detectors/rine+identity.yaml").read_text())
    assert raw["args"]["checkpoint"] is None


def test_grace_and_grace_d_share_an_adapter():
    """Stage 2 never touches the adapter, so the two variants must name the same
    checkpoint -- otherwise the comparison confounds two changes."""
    grace = yaml.safe_load((CONFIGS / "detectors/rine+grace.yaml").read_text())
    graced = yaml.safe_load((CONFIGS / "detectors/rine+grace-d.yaml").read_text())
    assert grace["args"]["checkpoint"] == graced["args"]["checkpoint"]
    assert grace["args"]["discrepancy"] is None
    assert graced["args"]["discrepancy"] is not None


def test_arms_differ_only_in_target_view():
    """Arm A and arm B must be one key apart, or the ablation is not an ablation."""
    a = yaml.safe_load((CONFIGS / "train/rine_degraded.yaml").read_text())
    b = yaml.safe_load((CONFIGS / "train/rine_clean.yaml").read_text())
    ignored = {"run_id", "target_view", "checkpoint_every", "loss"}
    assert {k: v for k, v in a.items() if k not in ignored} == {
        k: v for k, v in b.items() if k not in ignored
    }
    assert (a["target_view"], b["target_view"]) == ("degraded", "clean")


def test_defaults_yaml_is_documentation_only():
    """It concatenates every config kind, so it must never be loadable as one."""
    raw = yaml.safe_load((CONFIGS / "defaults.yaml").read_text())
    assert "run_id" in raw and "n_epochs" in raw and "adapter_checkpoint" in raw
