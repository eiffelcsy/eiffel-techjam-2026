"""The manifest: one table that every downstream component keys off.

Columns
-------
path        : str   absolute path on disk
label       : int   0 = real, 1 = generated
generator   : str   "flux1.1-pro" | "REAL" | ...
split       : str   "train" | "val"

Build it once, never reorder it (row order is the index used everywhere else).
"""

from pathlib import Path

import pandas as pd

COLUMNS = ["path", "label", "generator", "split"]


def build_manifest(source, out_path: str | Path) -> pd.DataFrame:
    """Materialize a source next to `out_path` and write its parquet manifest.

    Images land in `out_path.parent/images/`, so one directory holds a dataset
    and the table describing it.
    """
    out_path = Path(out_path)
    df = pd.DataFrame(list(source.rows(out_path.parent)), columns=COLUMNS)
    if df.empty:
        raise ValueError(f"source {type(source).__name__} produced no rows")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_manifest(path: str | Path, split: str | None = None) -> pd.DataFrame:
    """Load the manifest, optionally filtered to one split.

    The index is left as the original row number throughout: it is the image
    identity that seeds every degradation, so it must survive filtering and
    subsetting unchanged.
    """
    df = pd.read_parquet(path)
    if split is not None:
        df = df[df["split"] == split]
        if df.empty:
            raise ValueError(f"manifest has no rows with split={split!r}")
    return df


def sample_eval_subset(df: pd.DataFrame, n_per_class: int | None, seed: int) -> pd.DataFrame:
    """Fix the eval set once. Every reported number uses this set.

    Deterministic in `seed`, so "sampled once and cached" holds by
    reproducibility rather than by a cache file on disk. Row order is restored
    to manifest order afterwards, keeping conditions paired image-for-image.
    """
    if n_per_class is None:
        return df
    picked = [
        g.sample(n=min(n_per_class, len(g)), random_state=seed)
        for _, g in df.groupby("label", sort=True)
    ]
    return pd.concat(picked).sort_index()
