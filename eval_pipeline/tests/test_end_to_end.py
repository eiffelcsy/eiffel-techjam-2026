"""The whole harness on synthetic data: source -> manifest -> 21 conditions ->
run_eval -> report.

Hermetic by design: no network, no weights, no Hub. What it proves is that the
pieces compose and that the result JSON matches the schema `report.py` reads --
plus the two numbers that are analytic rather than empirical (clean AUC and L0
retention), which is where a polarity flip or a broken pairing would show up.
"""

import pytest

from pipeline.config import DatasetConfig, DegradeConfig, DetectorConfig, RunConfig
from pipeline.data.manifest import build_manifest, load_manifest
from pipeline.degrade.conditions import build_conditions, load_grid
from pipeline.eval.report import headline_table, load_results
from pipeline.eval.runner import run_eval
from tests.fixtures import SyntheticSource

GRID_FILE = "configs/degradations.yaml"
N_PER_CLASS, N_REPLICATES = 8, 2


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """Run the full pipeline once; every assertion below reads this."""
    root = tmp_path_factory.mktemp("run")
    manifest_path = root / "data" / "manifest.parquet"
    build_manifest(SyntheticSource(n_per_class=N_PER_CLASS), manifest_path)

    cfg = RunConfig(
        run_id="test",
        detector=DetectorConfig(
            name="stub", target="tests.fixtures.StubDetector", args={}, device="cpu",
        ),
        datasets=[DatasetConfig(
            name="synthetic", manifest=str(manifest_path), split="val",
        )],
        out_dir=str(root / "results"),
        batch_size=8,
        num_workers=0,
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
    assert len(by_level[1]) == sum(len(v) for v in grid.values()) == 14
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

    for name, entry in payload["levels"].items():
        assert {"auc", "retention", "errors"} <= set(entry)
        assert {"fpr", "fnr", "tp", "fp", "tn", "fn"} <= set(entry["errors"])
        if name != "L0_clean":
            assert len(entry["auc_ci"]) == 2
        if name in ("L2_pair", "L3_multi"):
            assert {"predicted_retention", "interaction_gap"} <= set(entry)

    assert len(payload["conditions"]) == 14
    for key, entry in payload["conditions"].items():
        assert key == f"{entry['transform']}/{entry['param_name']}={entry['param']}"
        assert {"group", "auc", "retention", "score_shift", "errors"} <= set(entry)

    assert len(payload["by_transform"]) == 6
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
