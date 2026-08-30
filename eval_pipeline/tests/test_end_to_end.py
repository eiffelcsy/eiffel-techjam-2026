"""The whole harness on synthetic data: source -> manifest -> 21 conditions ->
run_eval -> report.

Hermetic by design: no network, no weights, no Hub. What it proves is that the
pieces compose and that the result JSON matches the schema `report.py` reads --
plus the two numbers that are analytic rather than empirical (clean AUC and L0
retention), which is where a polarity flip or a broken pairing would show up.
"""

from collections import Counter
from pathlib import Path

import pytest

from pipeline.config import (
    DatasetConfig, DegradeConfig, DetectorConfig, RunConfig, load_run_config,
)
from pipeline.data.manifest import build_manifest, load_manifest
from pipeline.degrade.conditions import build_conditions, load_grid
from pipeline.detectors import build_detector
from pipeline.eval.report import headline_table, load_results
from pipeline.eval.runner import run_eval
from tests.fixtures import SyntheticSource

GRID_FILE = "configs/degradations.yaml"
N_PER_CLASS, N_REPLICATES = 8, 2
DETECTOR_CONFIGS = sorted(Path("configs/detectors").glob("*.yaml"))
DATASET_CONFIGS = sorted(Path("configs/datasets").glob("*.yaml"))


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


@pytest.mark.parametrize("path", DETECTOR_CONFIGS, ids=lambda p: p.stem)
def test_detector_configs_load(path):
    """Every shipped detector config parses and names an importable target.

    Not that it runs -- that needs weights this suite deliberately does not
    have -- only that the config is well-formed and the module resolves.
    """
    from pipeline.config import load_detector_config
    from common.imports import locate

    cfg = load_detector_config(path)
    assert cfg.name and cfg.device
    assert callable(locate(cfg.target))


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.stem)
def test_dataset_configs_load(path):
    """Every shipped dataset config parses and names an importable source.

    Not that it builds -- that needs the data, which this suite deliberately
    does not have -- only that the spec is well-formed and the target resolves.
    A `source:` block is optional: a config that only names a manifest built by
    another config (wildfake_train_val) is a legitimate spec with nothing to import.
    """
    from pipeline.config import load_dataset_config
    from common.imports import locate

    cfg = load_dataset_config(path)
    assert cfg.name and cfg.manifest
    if cfg.source is not None:
        assert callable(locate(cfg.source["target"]))


def test_csv_metadata_source_selects_by_path_and_labels_by_column(tmp_path):
    """The subsetting a table's own columns cannot express.

    WildFake is the case this exists for: every COCO image carries the same
    `Architecture` whichever 2017 directory it came from, so only the path
    separates the val2017 subset -- and the row that must NOT be picked up here
    is the one that matches on every column and differs only in its directory.
    """
    import csv

    from PIL import Image
    from pipeline.data.sources import CsvMetadataSource

    root = tmp_path / "images"
    rows = [
        ("./real/val2017/a.jpg", "0"),      # wanted, real
        ("./real/val2017/b.jpg", "0"),      # wanted, real
        ("./real/train2017/c.jpg", "0"),    # same columns, wrong directory
        ("./gen/Advanced/d.jpg", "1"),      # wanted, fake
        ("./gen/Typical/e.jpg", "1"),       # older tier of the same generator
    ]
    for rel, _ in rows:
        path = root / rel.removeprefix("./")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path, "JPEG")

    table = tmp_path / "test_metadata.csv"
    with table.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image_path", "IsFake"])
        w.writerows(rows)

    def source(**kw):
        return CsvMetadataSource(
            csv_path=str(table), root=str(root),
            path_column="Image_path", label_column="IsFake",
            fake_values=["1"], real_values=["0"],
            path_prefix=["real/val2017/", "gen/Advanced/"],
            generator="gen-v3", split="test", **kw,
        )

    got = list(source().rows(tmp_path / "unused"))
    assert [Path(r["path"]).name for r in got] == ["a.jpg", "b.jpg", "d.jpg"]
    assert [r["label"] for r in got] == [0, 0, 1]
    assert [r["generator"] for r in got] == ["REAL", "REAL", "gen-v3"]
    assert {r["split"] for r in got} == {"test"}

    # `limit` is per class, as on the Hub source.
    assert [r["label"] for r in source(limit=1).rows(tmp_path / "unused")] == [0, 1]

    # Referenced in place, never re-encoded: a JPEG benchmark rewritten as PNG
    # would be measuring the resave, not the generator.
    assert all(Path(r["path"]).is_file() for r in got)
    assert not (tmp_path / "unused").exists()

    # A file the table names but disk does not have is loud by default --
    # a half-unpacked archive is otherwise a silently shrunken benchmark.
    (root / "real" / "val2017" / "a.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="not on disk"):
        list(source().rows(tmp_path / "unused"))
    kept = list(source(on_missing="skip").rows(tmp_path / "unused"))
    assert [Path(r["path"]).name for r in kept] == ["b.jpg", "d.jpg"]

    # ...and skipping is still not allowed to hollow out a class: polarity is
    # checked over what survived, so a subset skipped down to nothing fails.
    (root / "real" / "val2017" / "b.jpg").unlink()
    with pytest.raises(ValueError, match="no real images"):
        list(source(on_missing="skip").rows(tmp_path / "unused"))


def _stratified_table(tmp_path):
    """A miniature WildFake: two real corpora, two generators, two held out.

    Group sizes are chosen so every proportional share is a whole number, so a
    rounding change shows up as a test failure rather than as a plausible
    off-by-one.
    """
    import csv

    from PIL import Image

    root = tmp_path / "images"
    rows = []                                      # (path, IsFake, Architecture)
    for i in range(12):
        rows.append((f"./Real/big/{i}.jpg", "0", "bigreal"))
    for i in range(4):
        rows.append((f"./Real/small/{i}.jpg", "0", "smallreal"))
    for i in range(19):
        rows.append((f"./Gen/A/{i}.jpg", "1", "A"))
    # A CJK filename, as Stable Diffusion's lora/ tree carries: the tables are
    # UTF-8 and the Windows locale codec cannot decode them.
    rows.append(("./Gen/A/拷贝.jpg", "1", "A"))
    for i in range(4):
        rows.append((f"./Gen/B/{i}.jpg", "1", "B"))
    # The held-out strata -- the evaluation set's own, never to be sampled.
    for i in range(5):
        rows.append((f"./Gen/HELD/{i}.jpg", "1", "heldgen"))
    for i in range(5):
        rows.append((f"./Real/held/{i}.jpg", "0", "heldreal"))

    for rel, _, _ in rows:
        path = root / rel.removeprefix("./")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path, "JPEG")

    table = tmp_path / "metadata.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_path", "IsFake", "Architecture"])
        w.writerows(rows)
    return table, root, rows


def test_stratified_source_samples_every_group_in_proportion(tmp_path):
    """`limit` takes a contiguous block; this takes a spread of known shape.

    The pools are 12:4 real and 20:4 fake, so at n_total=10 with a 0.4 real
    fraction the only correct answer is 3+1 real and 5+1 fake.
    """
    from pipeline.data.sources import StratifiedCsvSource

    table, root, rows = _stratified_table(tmp_path)

    def source(**kw):
        kw.setdefault("exclude_groups", ["heldgen", "heldreal"])
        return StratifiedCsvSource(
            csv_paths=str(table), root=str(root),
            path_column="Image_path", label_column="IsFake",
            fake_values=["1"], real_values=["0"],
            group_column="Architecture", generator_column="Architecture",
            n_total=10, real_fraction=0.4, split="train", **kw,
        )

    got = list(source().rows(tmp_path / "unused"))
    assert len(got) == 10
    by_group = Counter(r["generator"] for r in got)
    assert by_group == Counter({"A": 5, "B": 1, "REAL": 4})
    assert Counter(r["label"] for r in got) == Counter({0: 4, 1: 6})

    # The held-out strata are absent, which is the property the whole class
    # exists to guarantee.
    assert not any("HELD" in r["path"] or "held" in Path(r["path"]).parts[-2]
                   for r in got)

    # Real corpora split 3:1 the way their pools do.
    real_dirs = Counter(Path(r["path"]).parts[-2] for r in got if r["label"] == 0)
    assert real_dirs == Counter({"big": 3, "small": 1})

    # Table order is preserved -- the manifest index seeds every degradation.
    order = [r.removeprefix("./") for r, _, _ in rows]
    assert [str(Path(r["path"]).relative_to(root)).replace("\\", "/") for r in got] \
        == sorted((str(Path(r["path"]).relative_to(root)).replace("\\", "/") for r in got),
                  key=order.index)

    # Deterministic across runs, and actually responsive to the seed.
    assert [r["path"] for r in source().rows(tmp_path / "u")] == [r["path"] for r in got]
    assert [r["path"] for r in source(seed=7).rows(tmp_path / "u")] != [r["path"] for r in got]

    # Referenced in place, never re-encoded, like every CSV-backed source.
    assert not (tmp_path / "unused").exists()


def test_stratified_source_refuses_an_unmatched_exclusion(tmp_path):
    """The leakage guard's own failure mode.

    A misspelt held-out group does not raise anywhere downstream -- it just
    quietly trains on the benchmark's data -- so it has to raise here.
    """
    from pipeline.data.sources import StratifiedCsvSource

    table, root, _ = _stratified_table(tmp_path)

    def source(exclude):
        return StratifiedCsvSource(
            csv_paths=str(table), root=str(root),
            path_column="Image_path", label_column="IsFake",
            fake_values=["1"], real_values=["0"],
            group_column="Architecture", n_total=10,
            exclude_groups=exclude,
        )

    with pytest.raises(ValueError, match="matched no row"):
        list(source(["heldgen", "heldreal", "HeldGen"]).rows(tmp_path / "u"))

    # Excluding a whole class is refused too, rather than yielding a manifest
    # with nothing to score against.
    with pytest.raises(ValueError, match="no real rows"):
        list(source(["bigreal", "smallreal", "heldreal"]).rows(tmp_path / "u"))

    # A file the table names but disk lacks stays loud: silently dropping it
    # would re-weight every remaining stratum.
    (root / "Gen" / "A" / "0.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="not on disk"):
        list(source(["heldgen", "heldreal"]).rows(tmp_path / "u"))


def test_run_config_takes_one_detector_or_many(tmp_path):
    """`detector:` and `detectors:` are the same key at different arities."""
    def write(body: str) -> Path:
        path = tmp_path / "run.yaml"
        path.write_text(
            f"run_id: r\ndatasets: [configs/datasets/wildfake_train.yaml]\n{body}"
        )
        return path

    one = load_run_config(write("detector: configs/detectors/dinov3-wildfake.yaml\n"))
    assert [d.name for d in one.detectors] == ["dinov3-wildfake"]

    many = load_run_config(write(
        "detectors:\n"
        "  - configs/detectors/dinov3-wildfake.yaml\n"
        "  - configs/detectors/dinov3-wildfake-crop.yaml\n"
    ))
    assert [d.name for d in many.detectors] == ["dinov3-wildfake", "dinov3-wildfake-crop"]

    with pytest.raises(KeyError):
        load_run_config(write("detector: x\ndetectors: [y]\n"))
    with pytest.raises(KeyError):
        load_run_config(write(""))
