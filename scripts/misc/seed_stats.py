"""Aggregate repeated runs into mean / std / CI, and say whether a gap is real.

Every number in this project came from `seed: 0`. That is fine for a
deterministic reproduction check -- v1.2 reproduced an earlier run to 3e-5 --
but it says nothing about how much a metric moves when only the seed changes,
and without that a sweep cannot be read. The v1.1-v1.4 geometry sweep spanned
0.0009 AUC on a set whose marginal standard error is 0.0081, and picking a
winner from it was reading noise.

This script measures that noise floor directly:

    python scripts/seed_stats.py dinov3_multiscale_final_s*     # one group
    python scripts/seed_stats.py --vs dinov3_sweep_wratio_1     # against a run

The floor it reports is *seed* variance at a fixed configuration, which is the
right yardstick for comparing configurations trained on the same data. It is
narrower than the Hanley-McNeil sampling error of the val set itself (0.0081 on
wildfake-train-val), which is the right yardstick for the absolute number you
finally report. Both matter, and they answer different questions:

    seed spread   -- "is configuration A better than configuration B?"
    sampling SE   -- "how precise is the number I am publishing?"

A difference smaller than the seed spread is not evidence of anything.
"""

import argparse
import glob as globmod
import json
import statistics as st
from pathlib import Path

CKPT = Path("checkpoints/grace")
AXES = ["held_out_degradations", "held_out_images/wildfake-train-val"]
METRICS = ["auc_adapted", "acc_adapted", "retention", "cosine_to_clean"]


def final_row(summary: dict, axis: str) -> dict:
    """The finished adapter's row for one axis -- the highest val epoch."""
    rows = summary["validation"][axis]
    key = sorted(rows, key=lambda k: int(k.split("_")[1]))[-1]
    return rows[key]


def load(run_ids: list[str]) -> dict[str, dict]:
    out = {}
    for rid in run_ids:
        p = CKPT / rid / "summary.json"
        if not p.exists():
            raise SystemExit(f"no summary.json for run {rid!r} (looked in {p})")
        out[rid] = json.loads(p.read_text(encoding="utf-8"))
    return out


def expand(patterns: list[str]) -> list[str]:
    """Glob against the checkpoint dir so shells that do not expand still work."""
    found = []
    for pat in patterns:
        hits = sorted(
            Path(p).name for p in globmod.glob(str(CKPT / pat))
            if (Path(p) / "summary.json").exists()
        )
        found.extend(hits or [pat])
    ordered = []
    for run in found:
        if run not in ordered:
            ordered.append(run)
    return ordered


def summarize(group: dict[str, dict], label: str) -> dict:
    print(f"\n{label}  (n={len(group)}: {', '.join(group)})")
    stats = {}
    for axis in AXES:
        print(f"  {axis}")
        for m in METRICS:
            vals = [final_row(s, axis)[m] for s in group.values()]
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            stats[(axis, m)] = (mean, sd, vals)
            rng = f"{min(vals):.6f}..{max(vals):.6f}" if len(vals) > 1 else "-"
            print(f"    {m:<16} mean={mean:.6f}  sd={sd:.6f}  range={rng}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run ids or globs under checkpoints/grace/")
    ap.add_argument("--vs", nargs="+", default=None, help="second group to compare against")
    args = ap.parse_args()

    a_ids = expand(args.runs)
    a = load(a_ids)
    a_stats = summarize(a, "GROUP A")

    if not args.vs:
        print("\nUse --vs <runs> to test a configuration against this floor.")
        return

    b_ids = expand(args.vs)
    b = load(b_ids)
    b_stats = summarize(b, "GROUP B")

    print("\nB - A, against A's seed spread")
    print(f"  {'axis / metric':<42} {'delta':>10} {'A sd':>9} {'|d|/sd':>8}  verdict")
    for axis in AXES:
        for m in METRICS:
            am, asd, _ = a_stats[(axis, m)]
            bm, _, _ = b_stats[(axis, m)]
            d = bm - am
            ratio = abs(d) / asd if asd > 0 else float("inf")
            # 2 sd is the usual bar; below 1 sd a difference is indistinguishable
            # from re-running the same config with a different seed.
            verdict = ("within seed noise" if ratio < 1
                       else "suggestive" if ratio < 2
                       else "outside seed noise")
            print(f"  {axis.split('/')[-1] + ' / ' + m:<42} {d:+10.6f} {asd:9.6f} {ratio:8.2f}  {verdict}")


if __name__ == "__main__":
    main()
