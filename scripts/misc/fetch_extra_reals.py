"""Fetch exactly the additional reals the combined manifest samples.

    python scripts/misc/fetch_extra_reals.py --dry-run     # plan, download nothing
    python scripts/misc/fetch_extra_reals.py               # do it
    python scripts/misc/fetch_extra_reals.py --resume      # skip archives already extracted

The combined manifest adds reals by sampling laion5b + imagenet DISJOINT from the
reals already in the base manifest. Those images were never downloaded -- the
original fetch mined only the base sample and deleted each archive afterwards --
so this re-downloads the two real archives and extracts just the members the
reals source plans, deleting each zip afterwards. Peak disk is one archive plus
the extracted set.

It reuses fetch_wildfake_train.py's download/index/extract machinery: the archive
rooting quirks are the same, and there is exactly one place they are handled. The
exclusion set comes from `load_data.manifest.manifest_rel_paths` -- the same
helper the combined-manifest builder uses -- so both plan against the same sample
and the manifest build never references an image this script did not place.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.misc.fetch_wildfake_train import (  # noqa: E402  (imports after sys.path)
    COMPLETE, IMAGES, download, extract, index, metadata_tails,
)

CONFIG = "load_data/configs/datasets/wildfake_train_combined.yaml"

REAL_ARCHIVES = {k: COMPLETE[k] for k in ("Real/laion5b/", "Real/imagenet/")}


def _planned_reals():
    """The combined config's reals source, and the relative paths it chose."""
    from common.imports import instantiate
    from common.io import read_yaml
    from load_data.manifest import load_manifest, manifest_rel_paths

    cfg = read_yaml(CONFIG)
    build = cfg["build"]
    base = load_manifest(build["base_manifest"])
    exclude = manifest_rel_paths(base, build["reals_root"], 0)

    spec = next(s for s in build["sources"] if s.get("exclude_from_base"))
    src = instantiate(spec, exclude_paths=exclude)

    chosen = src._plan()
    wanted = {
        row[src.path_column].strip().removeprefix("./")
        for i, row in src._reader()
        if i in chosen
    }
    return src, wanted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="plan only, download nothing")
    p.add_argument("--resume", action="store_true", help="skip archives already extracted")
    p.add_argument("--keep-zips", action="store_true", help="do not delete after extracting")
    args = p.parse_args()

    src, wanted = _planned_reals()
    wanted = {w for w in wanted if any(w.startswith(prefix) for prefix in REAL_ARCHIVES)}
    print(f"{len(wanted)} extra reals to fetch (source: {type(src).__name__})")
    for prefix in REAL_ARCHIVES:
        print(f"  {prefix}: {sum(1 for w in wanted if w.startswith(prefix))}")
    if args.dry_run:
        return

    IMAGES.mkdir(parents=True, exist_ok=True)
    for n, (prefix, member) in enumerate(REAL_ARCHIVES.items(), 1):
        head = f"[{n}/{len(REAL_ARCHIVES)}] {Path(member).name}"
        needed = {w for w in wanted if w.startswith(prefix)}
        if args.resume and needed and all((IMAGES / w).exists() for w in needed):
            print(f"{head}: all {len(needed)} present, skipping")
            continue
        print(f"{head}: {len(needed)} candidates")
        z = download(member)
        family = prefix.split("/")[0] + "/"        # "Real/"
        got = extract(z, index(z, {family}, metadata_tails(family)), needed)
        print(f"  extracted {got}")
        if not args.keep_zips:
            z.unlink()
            print(f"  removed {z.name}")

    have = sum(1 for w in wanted if (IMAGES / w).exists())
    print(f"\ndone: {have}/{len(wanted)} extra reals under {IMAGES}")
    if have < len(wanted):
        raise SystemExit("some reals are missing -- rerun with --resume")


if __name__ == "__main__":
    main()
