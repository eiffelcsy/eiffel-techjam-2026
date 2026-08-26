"""Turn results JSON into something a human reads in five seconds.

Four views, coarse to fine.

1. Headline -- one row per detector, one column per level. The whole project in
   one table:

    detector      clean   L1 single   L2 pair   L3 multi   retention @ L3
    clip_linear   0.94      0.79        0.68      0.57          0.16
    dinov2        0.97      0.92        0.88      0.81          0.66

2. Interaction -- measured L2/L3 retention against what the L1 marginals
   predict if transforms did not interact. A large negative gap is the finding:
   single-transform benchmarks overstate that detector's robustness.

    detector      L2 predicted   L2 measured   gap      L3 predicted   L3 measured   gap
    clip_linear      0.74           0.61      -0.13        0.55           0.29      -0.26
    dinov2           0.90           0.88      -0.02        0.79           0.74      -0.05

3. By transform (L1) -- detectors x the eleven transforms, cells are AUC at the
   harshest setting. Attribution: which single cause hurts. Degradation curves
   (AUC vs parameter, one panel per transform) are the visual version.

4. Error split -- FPR and FNR at the clean threshold, per level and per
   transform. Detectors degrade asymmetrically: compression tends to make
   generated images look real (FNR climbs), noise tends to make real images
   look generated (FPR climbs). AUC hides both, and the direction can flip
   between L1 and L3.

Plus the worst-recipe table: which transform combinations at L2/L3 actually
did the damage.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.degrade.ops import TRANSFORMS  # noqa: E402
from pipeline.eval.runner import LEVEL_KEYS  # noqa: E402
from pipeline.utils.io import read_json  # noqa: E402

LEVEL_OF = {name: level for level, name in LEVEL_KEYS.items()}

SERIES = ["detector", "dataset"]
"""What identifies one row / line / panel.

A run is one detector against a set of datasets, and a results directory may
hold several runs, so neither field alone is unique -- every grouping, index
and plot legend keys on the pair.
"""


def series_label(detector: str, dataset: str) -> str:
    return f"{detector} @ {dataset}"


def load_results(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every results/*.json into two tidy frames:

      conditions: one row per (detector, dataset, condition) -- level, transform,
                  param, auc, retention, score_shift, flattened error counts.
                  `is_level` marks the four pooled level rows apart from the
                  19 individual L1 grid points.
      recipes:    one row per (detector, dataset, level, transform set)
    """
    condition_rows, recipe_rows = [], []

    for path in sorted(Path(results_dir).glob("*.json")):
        result = read_json(path)
        if "levels" not in result:          # not a run output
            continue
        detector = result["detector"]
        dataset = result.get("dataset") or "-"
        clean_auc = result["summary"]["clean_auc"]

        for name, entry in result["levels"].items():
            lo, hi = entry.get("auc_ci", [float("nan")] * 2)
            condition_rows.append({
                "detector": detector, "dataset": dataset,
                "condition": name, "level": LEVEL_OF[name],
                "is_level": True, "transform": None, "group": None,
                "param_name": None, "param": None,
                "auc": entry["auc"], "retention": entry["retention"],
                "score_shift": float("nan"),
                "auc_ci_lo": lo, "auc_ci_hi": hi,
                "predicted_retention": entry.get("predicted_retention", float("nan")),
                "interaction_gap": entry.get("interaction_gap", float("nan")),
                "clean_auc": clean_auc,
                **entry["errors"],
            })

        for name, entry in result["conditions"].items():
            condition_rows.append({
                "detector": detector, "dataset": dataset,
                "condition": name, "level": 1,
                "is_level": False, "transform": entry["transform"],
                "group": entry["group"], "param_name": entry["param_name"],
                "param": entry["param"],
                "auc": entry["auc"], "retention": entry["retention"],
                "score_shift": entry["score_shift"],
                "auc_ci_lo": float("nan"), "auc_ci_hi": float("nan"),
                "predicted_retention": float("nan"),
                "interaction_gap": float("nan"),
                "clean_auc": clean_auc,
                **entry["errors"],
            })

        for level, rows in result["recipes"].items():
            for row in rows:
                recipe_rows.append({
                    "detector": detector, "dataset": dataset, "level": level,
                    "transforms": " + ".join(row["transforms"]), **{
                        k: v for k, v in row.items() if k != "transforms"
                    },
                })

    return pd.DataFrame(condition_rows), pd.DataFrame(recipe_rows)


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """View 1: detectors x levels, AUC + retention."""
    levels = df[df["is_level"]]
    out = levels.pivot(index=SERIES, columns="condition", values="auc")
    out = out.reindex(columns=[c for c in LEVEL_KEYS.values() if c in out.columns])
    deepest = out.columns[-1] if len(out.columns) else None
    if deepest is not None:
        ret = levels.pivot(index=SERIES, columns="condition", values="retention")
        out[f"retention @ {deepest}"] = ret[deepest]
    return out.round(3)


def interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """View 2: predicted vs measured composed retention, and the gap."""
    composed = df[df["is_level"] & df["level"].isin([2, 3])]
    if composed.empty:
        return pd.DataFrame()
    out = composed.pivot(
        index=SERIES, columns="condition",
        values=["predicted_retention", "retention", "interaction_gap"],
    )
    out.columns = [f"{level} {metric.replace('retention', 'measured')}"
                   if metric == "retention" else f"{level} {metric}"
                   for metric, level in out.columns]
    return out.reindex(sorted(out.columns), axis=1).round(3)


def by_transform_table(df: pd.DataFrame, at: str = "worst") -> pd.DataFrame:
    """View 3: detectors x transforms at L1. `at` is "worst" or "mean"."""
    grid = df[~df["is_level"]]
    if grid.empty:
        return pd.DataFrame()
    agg = "min" if at == "worst" else "mean"
    return (
        grid.groupby([*SERIES, "transform"])["auc"].agg(agg)
        .unstack("transform").round(3)
    )


def error_table(df: pd.DataFrame, by: str = "level") -> pd.DataFrame:
    """View 4: FPR / FNR / precision / recall. `by` is "level" or "transform"."""
    cols = ["fpr", "fnr", "precision", "recall"]
    if by == "level":
        rows = df[df["is_level"]]
        return rows.set_index([*SERIES, "condition"])[cols].round(3)
    rows = df[~df["is_level"]]
    return rows.groupby([*SERIES, "transform"])[cols].mean().round(3)


def worst_recipes_table(recipes: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The k most damaging transform combinations per detector."""
    if recipes.empty:
        return pd.DataFrame()
    return (
        recipes.sort_values("auc")
        .groupby(SERIES, group_keys=False)
        .head(k)
        .set_index([*SERIES, "level", "transforms"])[["n", "auc", "retention", "fpr", "fnr"]]
        .round(3)
    )


def plot_level_curve(df: pd.DataFrame, out_path: str) -> None:
    """AUC vs composition level (0..3), one line per detector, with the CI band
    on the composed levels. The single figure to put on a slide."""
    levels = df[df["is_level"]].sort_values("level")
    fig, ax = plt.subplots(figsize=(6, 4))
    for (detector, dataset), rows in levels.groupby(SERIES):
        ax.plot(rows["level"], rows["auc"], marker="o", label=series_label(detector, dataset))
        ax.fill_between(rows["level"], rows["auc_ci_lo"], rows["auc_ci_hi"], alpha=0.15)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.annotate("chance", (0.02, 0.505), xycoords=("axes fraction", "data"),
                fontsize=8, color="grey")
    ax.set_xticks(sorted(levels["level"].unique()))
    ax.set_xticklabels([LEVEL_KEYS[i] for i in sorted(levels["level"].unique())])
    ax.set_ylabel("AUC")
    ax.set_xlabel("composition level")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_degradation_curves(df: pd.DataFrame, out_path: str) -> None:
    """L1 small multiples: one panel per transform, x = parameter (mild ->
    severe), y = AUC, one line per detector, dashed line at clean AUC."""
    grid = df[~df["is_level"]]
    names = [t for t in TRANSFORMS if t in set(grid["transform"])]
    if not names:
        return
    ncols = min(3, len(names))
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for ax, name in zip(axes.flat, names):
        panel = grid[grid["transform"] == name]
        order = {p: i for i, p in enumerate(TRANSFORMS[name].params)}
        for (detector, dataset), rows in panel.groupby(SERIES):
            rows = rows.assign(_o=rows["param"].map(order)).sort_values("_o")
            ax.plot(rows["_o"], rows["auc"], marker="o", label=series_label(detector, dataset))
            ax.axhline(rows["clean_auc"].iloc[0], ls="--", lw=1, alpha=0.4)
        ax.set_xticks(list(order.values()))
        ax.set_xticklabels([str(p) for p in order])
        ax.set_title(f"{name}\n{TRANSFORMS[name].analog}", fontsize=9)
        ax.set_xlabel(TRANSFORMS[name].param_name)
        ax.set_ylabel("AUC")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    _save(fig, out_path)


def plot_error_split(df: pd.DataFrame, out_path: str) -> None:
    """Stacked FP/FN bars per level -- shows the direction of failure and
    whether it flips as compositions deepen."""
    levels = df[df["is_level"]].sort_values([*SERIES, "level"])
    series = sorted(levels.groupby(SERIES).groups)
    fig, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 3.5),
                             squeeze=False, sharey=True)

    for ax, key in zip(axes.flat, series):
        rows = levels[(levels["detector"] == key[0]) & (levels["dataset"] == key[1])]
        x = range(len(rows))
        ax.bar(x, rows["fpr"], label="FPR (real called AI)")
        ax.bar(x, rows["fnr"], bottom=rows["fpr"], label="FNR (AI called real)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([LEVEL_KEYS[i] for i in rows["level"]], rotation=45, ha="right")
        ax.set_title(series_label(*key), fontsize=9)
    axes.flat[0].set_ylabel("error rate @ clean threshold")
    axes.flat[0].legend(frameon=False, fontsize=8)
    _save(fig, out_path)


def render_markdown(conditions: pd.DataFrame, recipes: pd.DataFrame, out_path: str) -> None:
    """Write the four tables plus the three figures into one summary.md."""
    out_path = Path(out_path)
    out_dir = out_path if out_path.suffix == "" else out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "summary.md"

    figures = {
        "level_curve.png": plot_level_curve,
        "degradation_curves.png": plot_degradation_curves,
        "error_split.png": plot_error_split,
    }
    for filename, plot in figures.items():
        plot(conditions, str(out_dir / filename))

    sections = [
        ("Headline -- AUC by composition level", headline_table(conditions)),
        ("Interaction -- measured vs. independence prediction", interaction_table(conditions)),
        ("By transform (L1, worst parameter)", by_transform_table(conditions, at="worst")),
        ("Error split by level", error_table(conditions, by="level")),
        ("Worst recipes (L2/L3)", worst_recipes_table(recipes)),
    ]

    parts = ["# Robustness evaluation summary\n"]
    for title, table in sections:
        parts.append(f"\n## {title}\n")
        parts.append("_no data_\n" if table.empty else table.to_markdown() + "\n")
    parts.append("\n## Figures\n")
    parts.extend(f"\n![{name}]({name})\n" for name in figures)

    md_path.write_text("".join(parts))


def _save(fig, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
