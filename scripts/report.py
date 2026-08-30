"""Aggregate results/*.json into the summary tables + figures.

    python scripts/report.py --results results/ --out results/summary
"""

import argparse

from eval.report import (
    by_transform_table, headline_table, interaction_table, load_results,
    render_markdown, worst_recipes_table,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/")
    p.add_argument("--out", default="results/summary")
    args = p.parse_args()

    conditions, recipes = load_results(args.results)
    if conditions.empty:
        raise SystemExit(f"no result JSONs found under {args.results!r}")

    render_markdown(conditions, recipes, args.out)

    for title, table in [
        ("Headline", headline_table(conditions)),
        ("Interaction", interaction_table(conditions)),
        ("By transform (worst)", by_transform_table(conditions)),
        ("Worst recipes", worst_recipes_table(recipes, k=5)),
    ]:
        print(f"\n== {title} ==")
        print("(no data)" if table.empty else table.to_string())

    print(f"\nwrote {args.out}/summary.md and three figures")


if __name__ == "__main__":
    main()
