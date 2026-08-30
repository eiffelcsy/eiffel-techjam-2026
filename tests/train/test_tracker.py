"""W&B tracking: off by default, never load-bearing, never fatal.

The point of these tests is the *absence* of behaviour. A tracker is wired into
all three training entry points, and the thing that must be true of it is that a
machine with no `wandb` installed, no login and no network runs exactly the same
training and writes exactly the same files as one with all three.

`wandb` is not a dependency of this package, so the live path is exercised
against a stub module injected into `sys.modules` -- which also lets the failure
paths (a raising `log`, a raising `init`) be tested at all, since a real client
succeeds.
"""

import argparse
import sys
import types
import warnings

import pytest

from train.config import TrainConfig, WandbConfig
from train.tracker import (
    NullTracker, add_wandb_args, apply_wandb_args, build_tracker, flatten_config,
)


@pytest.fixture
def stub_wandb(monkeypatch):
    """A minimal `wandb` module recording what it was told."""
    calls = {"init": [], "log": [], "summary": {}, "finished": False}

    class _Run:
        url = "https://wandb.test/run/1"
        summary = calls["summary"]

        def log(self, metrics, step=None):
            calls["log"].append((dict(metrics), step))

        def finish(self):
            calls["finished"] = True

    module = types.ModuleType("wandb")
    module.init = lambda **kw: (calls["init"].append(kw), _Run())[1]
    monkeypatch.setitem(sys.modules, "wandb", module)
    return calls, module


def _cfg(**kw):
    return WandbConfig(**{"enabled": True, "project": "p", **kw})


# ------------------------------------------------------------------- off ----

def test_disabled_by_default():
    assert TrainConfig(run_id="r", cache_dir="c").wandb.enabled is False


@pytest.mark.parametrize("cfg", [None, WandbConfig(), WandbConfig(enabled=False)])
def test_off_gives_a_null_tracker_that_swallows_everything(cfg):
    tracker = build_tracker(cfg, run_id="r", job_type="stage1")
    assert isinstance(tracker, NullTracker) and tracker.enabled is False
    tracker.log({"loss": 1.0}, step=3)
    tracker.summary({"auc": 0.9})
    tracker.finish()


def test_enabled_without_the_package_is_a_configuration_error(monkeypatch):
    """Raised, not warned: the config asked for tracking in so many words, and
    silently not tracking a sweep is worse than failing at second zero."""
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="pip install wandb"):
        build_tracker(_cfg(), run_id="r", job_type="stage1")


# -------------------------------------------------------------------- on ----

def test_enabled_starts_a_run_named_by_run_id(stub_wandb):
    calls, _ = stub_wandb
    tracker = build_tracker(
        _cfg(group="e4", tags=["poc"]), run_id="dinov3_clean",
        job_type="stage1", config={"loss": {"lam_kl": 0.1}},
    )
    (kw,) = calls["init"]
    assert kw["name"] == "dinov3_clean" and kw["group"] == "e4"
    assert kw["job_type"] == "stage1" and kw["tags"] == ["poc"]
    assert kw["config"] == {"loss/lam_kl": 0.1}
    assert tracker.enabled and tracker.url


def test_step_is_explicit_and_never_logged_as_a_metric(stub_wandb):
    """W&B's implicit counter would interleave the 50-step diagnostics with the
    end-of-epoch rows; and a `step` metric plots a diagonal line."""
    calls, _ = stub_wandb
    tracker = build_tracker(_cfg(), run_id="r", job_type="stage1")
    tracker.log({"loss": 0.5, "step": 100}, step=100)
    assert calls["log"] == [({"loss": 0.5}, 100)]


def test_summary_is_flattened_so_the_runs_table_can_sort_it(stub_wandb):
    calls, _ = stub_wandb
    tracker = build_tracker(_cfg(), run_id="r", job_type="stage2")
    tracker.summary({"validation": {"epoch_10000": {"auc_aux": 0.61}}, "beta": 0.2})
    assert calls["summary"] == {"validation/epoch_10000/auc_aux": 0.61, "beta": 0.2}


def test_finish_is_forwarded(stub_wandb):
    calls, _ = stub_wandb
    build_tracker(_cfg(), run_id="r", job_type="stage1").finish()
    assert calls["finished"] is True


# --------------------------------------------------------------- failures ----

def test_a_failing_init_warns_and_downgrades(monkeypatch):
    """Auth and network failures are transient; the training run is still worth
    doing, and its summary.json is unaffected."""
    module = types.ModuleType("wandb")

    def _boom(**kw):
        raise RuntimeError("no api key")

    module.init = _boom
    monkeypatch.setitem(sys.modules, "wandb", module)

    with pytest.warns(UserWarning, match="continuing untracked"):
        tracker = build_tracker(_cfg(), run_id="r", job_type="stage1")
    assert isinstance(tracker, NullTracker)


def test_a_failing_log_warns_once_then_goes_quiet(stub_wandb):
    """A dead network mid-run must not raise into the training loop, and must
    not warn on every one of the next ten thousand steps."""
    _, module = stub_wandb

    class _Run:
        summary = {}

        def log(self, metrics, step=None):
            raise ConnectionError("gone")

        def finish(self):
            pass

    module.init = lambda **kw: _Run()
    tracker = build_tracker(_cfg(), run_id="r", job_type="stage1")

    with pytest.warns(UserWarning, match="continuing untracked"):
        tracker.log({"loss": 1.0}, step=0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")      # a second warning would now raise
        tracker.log({"loss": 1.0}, step=1)
        tracker.summary({"auc": 0.5})
        tracker.finish()


# ----------------------------------------------------------------- config ----

def test_flatten_config_takes_a_dataclass():
    flat = flatten_config(TrainConfig(run_id="r", cache_dir="c"))
    assert flat["run_id"] == "r"
    assert flat["loss/lam_kl"] == 0.1          # nested dataclass, flattened
    assert flat["adapter/bottleneck"] == 256
    assert "loss" not in flat                  # and not also present as a blob


# -------------------------------------------------------------------- CLI ----

def _parse(argv):
    p = argparse.ArgumentParser()
    add_wandb_args(p)
    return p.parse_args(argv)


def test_absent_flags_do_not_switch_off_a_config_that_enabled_tracking():
    cfg = TrainConfig(run_id="r", cache_dir="c", wandb=WandbConfig(enabled=True))
    apply_wandb_args(cfg, _parse([]))
    assert cfg.wandb.enabled is True


def test_no_wandb_overrides_a_config_that_enabled_tracking():
    cfg = TrainConfig(run_id="r", cache_dir="c", wandb=WandbConfig(enabled=True))
    apply_wandb_args(cfg, _parse(["--no-wandb"]))
    assert cfg.wandb.enabled is False


@pytest.mark.parametrize(
    "argv,field,value",
    [
        (["--wandb-project", "grace"], "project", "grace"),
        (["--wandb-group", "e4_erasure"], "group", "e4_erasure"),
        (["--wandb-offline"], "mode", "offline"),
    ],
)
def test_naming_a_project_group_or_mode_implies_tracking(argv, field, value):
    """`--wandb-group e4_erasure` without `--wandb` is not a request to log
    nothing."""
    cfg = TrainConfig(run_id="r", cache_dir="c")
    apply_wandb_args(cfg, _parse(argv))
    assert cfg.wandb.enabled is True and getattr(cfg.wandb, field) == value


def test_explicit_no_wandb_beats_an_implication():
    cfg = TrainConfig(run_id="r", cache_dir="c")
    apply_wandb_args(cfg, _parse(["--wandb-group", "e4", "--no-wandb"]))
    assert cfg.wandb.enabled is False
