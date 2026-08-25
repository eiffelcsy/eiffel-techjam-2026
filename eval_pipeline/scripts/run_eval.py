"""THE deliverable: clean vs composed-degradation AUC for one detector.

    python scripts/run_eval.py --config configs/runs/<name>.yaml

A run is one detector against a set of datasets. Writes
results/{run_id}__{detector}__{dataset}.json per dataset and prints the
headline table over everything in out_dir.
"""

import argparse

from pipeline.config import load_run_config
from pipeline.eval.report import headline_table, load_results
from pipeline.eval.runner import run_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="run spec yaml")
    p.add_argument("--levels", nargs="+", type=int, default=None,
                   help="subset of [0,1,2,3]; smoke runs usually want 0 1")
    p.add_argument("--transforms", nargs="+", default=None,
                   help="subset of the six, e.g. --transforms jpeg resize")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    if args.levels is not None:
        cfg.degrade.levels = args.levels
    if args.transforms is not None:
        cfg.degrade.transforms = args.transforms
    if 0 not in cfg.degrade.levels:
        # L0 sets the threshold and the retention denominator for every other
        # level, so it is not optional even when only a subset is requested.
        cfg.degrade.levels = [0, *cfg.degrade.levels]

    run_eval(cfg)
    conditions, _ = load_results(cfg.out_dir)
    print(headline_table(conditions).to_string())


if __name__ == "__main__":
    main()
