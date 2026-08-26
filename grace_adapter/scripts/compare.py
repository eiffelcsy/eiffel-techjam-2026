"""Compare an adapted detector against its baseline on a shared denominator.

    python scripts/compare.py --baseline ../eval_pipeline/results/zoo__rine__ntire_val.json \
                              --adapted  ../eval_pipeline/results/grace__rine+grace__ntire_val.json

The harness normalizes retention by *each detector's own* clean AUC, which is the
right convention for describing one detector and the wrong one for comparing two.
GRACE-D changes both the numerator and the denominator: its auxiliary head reads
Δ, which is ~0 on clean images, so its clean AUC is roughly the baseline's while
its degraded AUC can be higher. Normalizing by its own clean AUC would hide
exactly the effect being claimed.

So this reports **baseline-normalized retention**:

    (auc_adapted_degraded − 0.5) / (auc_baseline_clean − 0.5)

which is > 1.0 exactly when the adapted detector, on a degraded image, beats the
baseline detector on the clean one. For GRACE that is impossible by construction:
restoration is bounded above by the clean-feature score. For GRACE-D it is the
headline claim, because the magnitude of the damage is information the clean
image does not contain.

Post-hoc and read-only. `eval_pipeline` is untouched, and both inputs are
ordinary harness result JSONs.
"""

import argparse
import json
from pathlib import Path

CHANCE = 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="harness result JSON for the base detector")
    p.add_argument("--adapted", required=True, help="harness result JSON for the adapted one")
    p.add_argument("--out", help="write the comparison as JSON")
    return p.parse_args()


def retention(auc: float, clean_auc: float) -> float:
    denom = clean_auc - CHANCE
    return float("nan") if abs(denom) < 1e-12 else (auc - CHANCE) / denom


def main():
    args = parse_args()
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    adapt = json.loads(Path(args.adapted).read_text(encoding="utf-8"))

    if base["dataset"] != adapt["dataset"]:
        raise SystemExit(
            f"different datasets ({base['dataset']} vs {adapt['dataset']}); "
            f"retention is only comparable on the same eval set"
        )

    clean = base["levels"]["L0_clean"]["auc"]
    rows = []
    for key in base["levels"]:
        b, a = base["levels"][key], adapt["levels"].get(key)
        if a is None:
            continue
        rows.append({
            "level": key,
            "auc_baseline": b["auc"],
            "auc_adapted": a["auc"],
            "retention_baseline": retention(b["auc"], clean),
            "retention_adapted": retention(a["auc"], clean),
            "delta": retention(a["auc"], clean) - retention(b["auc"], clean),
        })

    conditions = []
    for cond, b in base.get("conditions", {}).items():
        a = adapt.get("conditions", {}).get(cond)
        if a:
            conditions.append({
                "condition": cond,
                "delta": retention(a["auc"], clean) - retention(b["auc"], clean),
                "auc_baseline": b["auc"],
                "auc_adapted": a["auc"],
            })
    conditions.sort(key=lambda r: r["delta"], reverse=True)

    report = {
        "dataset": base["dataset"],
        "baseline": base["detector"],
        "adapted": adapt["detector"],
        "baseline_clean_auc": clean,
        "levels": rows,
        "best_conditions": conditions[:5],
        "worst_conditions": conditions[-5:],
        "exceeds_clean_ceiling": any(r["retention_adapted"] > 1.0 for r in rows[1:]),
    }

    width = max(len(r["level"]) for r in rows) if rows else 10
    print(f"{base['detector']} -> {adapt['detector']} on {base['dataset']}")
    print(f"denominator: baseline clean AUC = {clean:.4f}\n")
    print(f"{'level'.ljust(width)}  base_ret  adpt_ret     delta")
    for r in rows:
        print(
            f"{r['level'].ljust(width)}  {r['retention_baseline']:8.4f}  "
            f"{r['retention_adapted']:8.4f}  {r['delta']:+8.4f}"
        )
    if report["exceeds_clean_ceiling"]:
        print("\nretention > 1.0: the adapted detector beats the baseline's CLEAN score")
        print("on degraded images. Only the discrepancy branch can do this -- check")
        print("that this run is GRACE-D and not GRACE before reporting it.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
