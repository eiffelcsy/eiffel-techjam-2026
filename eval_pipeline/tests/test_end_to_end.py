"""The whole harness on synthetic data: source -> manifest -> 21 conditions ->
run_eval -> report.

Hermetic by design: no network, no weights, no Hub. What it proves is that the
pieces compose and that the result JSON matches the schema `report.py` reads --
plus the two numbers that are analytic rather than empirical (clean AUC and L0
retention), which is where a polarity flip or a broken pairing would show up.
"""

from pathlib import Path

import pytest

from pipeline.config import (
    DatasetConfig, DegradeConfig, DetectorConfig, RunConfig, load_run_config,
)
from pipeline.data.manifest import build_manifest, load_manifest
from pipeline.degrade.conditions import build_conditions, load_grid
from pipeline.detectors import build_detector
from pipeline.detectors._vendor import THIRD_PARTY
from pipeline.eval.report import headline_table, load_results
from pipeline.eval.runner import run_eval
from tests.fixtures import SyntheticSource

GRID_FILE = "configs/degradations.yaml"
N_PER_CLASS, N_REPLICATES = 8, 2
ZOO_CONFIGS = sorted(Path("configs/detectors").glob("*.yaml"))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """Run the full pipeline once; every assertion below reads this."""
    root = tmp_path_factory.mktemp("run")
    manifest_path = root / "data" / "manifest.parquet"
    build_manifest(SyntheticSource(n_per_class=N_PER_CLASS), manifest_path)

    cfg = RunConfig(
        run_id="test",
        detectors=[DetectorConfig(
            name="stub", target="tests.fixtures.StubDetector", args={}, device="cpu",
        )],
        datasets=[DatasetConfig(
            name="synthetic", manifest=str(manifest_path), split="val",
        )],
        out_dir=str(root / "results"),
        batch_size=8,
        # Workers, not 0: forking them is what a preprocess_fn() holding the
        # model would fail on, and every zoo adapter has to get that right.
        num_workers=2,
        degrade=DegradeConfig(
            grid_file=GRID_FILE, levels=[0, 1, 2, 3],
            n_replicates=N_REPLICATES, transforms=None, seed=0,
        ),
    )
    results = run_eval(cfg)
    assert len(results) == 1
    return results[0], root


def test_manifest_is_balanced_and_ordered(result):
    _, root = result
    df = load_manifest(root / "data" / "manifest.parquet", split="val")
    assert len(df) == 2 * N_PER_CLASS
    assert df["label"].value_counts().to_dict() == {0: N_PER_CLASS, 1: N_PER_CLASS}
    assert list(df.index) == sorted(df.index), "manifest order must never be shuffled"


def test_condition_lattice_is_complete():
    grid = load_grid(GRID_FILE)
    conditions = build_conditions(grid, [0, 1, 2, 3], n_replicates=N_REPLICATES)
    by_level = {}
    for c in conditions:
        by_level.setdefault(c.level, []).append(c)

    assert len(by_level[0]) == 1
    assert len(by_level[1]) == sum(len(v) for v in grid.values()) == 19
    assert len(by_level[2]) == len(by_level[3]) == N_REPLICATES


def test_clean_auc_and_retention_are_analytic(result):
    payload, _ = result
    # The planted signal is separable with a margin far wider than the sampling
    # noise, so anything but 1.0 means the pairing, the labels, or the score
    # polarity is wrong -- not that the "model" is weak.
    assert payload["summary"]["clean_auc"] == pytest.approx(1.0)
    assert payload["levels"]["L0_clean"]["retention"] == pytest.approx(1.0)
    assert payload["n_real"] == payload["n_fake"] == N_PER_CLASS


def test_result_matches_the_schema_report_reads(result):
    payload, _ = result
    assert set(payload) >= {
        "run_id", "detector", "dataset", "n_images", "n_real", "n_fake",
        "clean_threshold", "levels", "conditions", "by_transform", "recipes",
        "summary",
    }
    assert (payload["run_id"], payload["detector"], payload["dataset"]) == (
        "test", "stub", "synthetic",
    )
    assert set(payload["levels"]) == {"L0_clean", "L1_single", "L2_pair", "L3_multi"}
    assert payload["detector_spec"]["target"] == "tests.fixtures.StubDetector"

    for name, entry in payload["levels"].items():
        assert {"auc", "retention", "errors"} <= set(entry)
        assert {"fpr", "fnr", "tp", "fp", "tn", "fn"} <= set(entry["errors"])
        if name != "L0_clean":
            assert len(entry["auc_ci"]) == 2
        if name in ("L2_pair", "L3_multi"):
            assert {"predicted_retention", "interaction_gap"} <= set(entry)

    assert len(payload["conditions"]) == 19
    for key, entry in payload["conditions"].items():
        assert key == f"{entry['transform']}/{entry['param_name']}={entry['param']}"
        assert {"group", "auc", "retention", "score_shift", "errors"} <= set(entry)

    assert len(payload["by_transform"]) == 11
    assert set(payload["recipes"]) == {"L2", "L3"}
    assert {"clean_auc", "retention_by_level", "worst_condition", "worst_recipe",
            "operating_envelope"} <= set(payload["summary"])


def test_degradation_actually_degrades(result):
    payload, _ = result
    # Not a claim about magnitude -- only that the degraded images reach the
    # detector at all. A silently-bypassed condition would leave these equal.
    assert payload["levels"]["L3_multi"]["auc"] < payload["levels"]["L0_clean"]["auc"]
    assert any(e["score_shift"] != 0.0 for e in payload["conditions"].values())


def test_report_renders_from_the_results(result):
    _, root = result
    conditions, recipes = load_results(root / "results")
    assert not conditions.empty and not recipes.empty

    headline = headline_table(conditions)
    assert list(headline.index) == [("stub", "synthetic")]
    assert "L0_clean" in headline.columns and "L3_multi" in headline.columns

    from pipeline.eval.report import render_markdown

    out = root / "summary"
    render_markdown(conditions, recipes, out)
    assert (out / "summary.md").read_text().startswith("# Robustness evaluation summary")
    for figure in ("level_curve.png", "degradation_curves.png", "error_split.png"):
        assert (out / figure).stat().st_size > 0


def test_a_zoo_of_detectors_lands_in_one_comparable_table(result, tmp_path):
    """Several detectors in one run: one result file each, one row each.

    The headline table pivots on (detector, dataset, condition), so two zoo
    members sharing a name would collide rather than compare. This is the
    cheapest place to catch that.
    """
    _, root = result
    cfg = RunConfig(
        run_id="zoo",
        detectors=[
            DetectorConfig(name="stub-a", target="tests.fixtures.StubDetector",
                           args={}, device="cpu"),
            DetectorConfig(name="stub-b", target="tests.fixtures.StubDetector",
                           args={"scale": 5.0}, device="cpu"),
        ],
        datasets=[DatasetConfig(name="synthetic",
                                manifest=str(root / "data" / "manifest.parquet"),
                                split="val")],
        out_dir=str(tmp_path / "results"),
        batch_size=8,
        num_workers=0,
        degrade=DegradeConfig(grid_file=GRID_FILE, levels=[0, 1],
                              n_replicates=1, transforms=["jpeg"], seed=0),
    )
    results = run_eval(cfg)
    assert [r["detector"] for r in results] == ["stub-a", "stub-b"]

    conditions, _ = load_results(tmp_path / "results")
    headline = headline_table(conditions)
    assert list(headline.index) == [("stub-a", "synthetic"), ("stub-b", "synthetic")]


@pytest.mark.parametrize("path", ZOO_CONFIGS, ids=lambda p: p.stem)
def test_detector_configs_load(path):
    """Every shipped detector config parses and names an importable target.

    Not that it runs -- that needs weights this suite deliberately does not
    have -- only that the config is well-formed and the module resolves.
    """
    from pipeline.config import load_detector_config
    from pipeline.utils.imports import locate

    cfg = load_detector_config(path)
    assert cfg.name and cfg.device
    assert callable(locate(cfg.target))


@pytest.mark.skipif(THIRD_PARTY.is_dir(), reason="the zoo is cloned on this machine")
@pytest.mark.parametrize(
    "path", [p for p in ZOO_CONFIGS if p.stem != "sdxl"], ids=lambda p: p.stem
)
def test_missing_zoo_clone_says_how_to_fix_it(path):
    """A missing clone is the likeliest first-run failure; it must be actionable."""
    from pipeline.config import load_detector_config

    with pytest.raises(FileNotFoundError, match="git clone"):
        build_detector(load_detector_config(path))


def test_run_config_takes_one_detector_or_many(tmp_path):
    """`detector:` and `detectors:` are the same key at different arities."""
    def write(body: str) -> Path:
        path = tmp_path / "run.yaml"
        path.write_text(
            f"run_id: r\ndatasets: [configs/datasets/sid_set.yaml]\n{body}"
        )
        return path

    one = load_run_config(write("detector: configs/detectors/rine-ldm.yaml\n"))
    assert [d.name for d in one.detectors] == ["rine-ldm"]

    many = load_run_config(write(
        "detectors:\n"
        "  - configs/detectors/rine-ldm.yaml\n"
        "  - configs/detectors/rine-4class.yaml\n"
    ))
    assert [d.name for d in many.detectors] == ["rine-ldm", "rine-4class"]

    with pytest.raises(KeyError):
        load_run_config(write("detector: x\ndetectors: [y]\n"))
    with pytest.raises(KeyError):
        load_run_config(write(""))
