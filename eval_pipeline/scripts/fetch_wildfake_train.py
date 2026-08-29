"""Fetch exactly the images configs/datasets/wildfake_train.yaml samples.

    python scripts/fetch_wildfake_train.py --dry-run     # plan, download nothing
    python scripts/fetch_wildfake_train.py               # do it
    python scripts/fetch_wildfake_train.py --resume      # skip finished archives

WildFake is 1,199 GB. This manifest needs 100,000 of its 3.57M images. Unpacking
whole archives to keep 3% of them would need ~194 GB of disk that nothing ever
reads, so each archive is downloaded, opened, mined for just the members the
sample names, and deleted before the next one starts. Peak disk is one archive
plus the extracted images -- tens of GB rather than hundreds.

TWO PHASES, and the order matters
---------------------------------
Stable Diffusion (497 GB) and Midjourney (530 GB) are far too large to take
whole, so one part of each is fetched and those two strata are sampled from
inside it. Which rows that leaves is not knowable from the metadata -- only the
archive's own index says what it holds -- so phase 1 downloads those two parts
and writes their file lists, and phase 2 plans the sample against them.

The strata keep their FULL corpus weight either way: `StratifiedCsvSource`
allocates on how big a family is in WildFake and only draws from what is
available, so SD stays 35.5% of the fake half rather than collapsing to the
share of it that was downloaded. If a part turns out too small to cover its
share, the planner raises instead of quietly under-filling.

Resumability: `--resume` skips any archive whose extracted files are all present
and re-downloads nothing. The availability lists under `available/` are the
record of what phase 1 saw and are not regenerated once written -- rewriting
them would re-plan the sample against a different subset and invalidate the
manifest.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO = "hy2628982280/WildFake"

# metadata path prefix -> archive in the repo. Complete families: the archive
# holds every row of that prefix, so the sample can be planned before download.
COMPLETE = {
    "Real/laion5b/": "Images/Real/laion5b.zip",
    "Real/imagenet/": "Images/Real/imagenet.zip",
    "Real/church/": "Images/Real/church.zip",
    "Real/ffhq/": "Images/Real/ffhq.zip",
    "Real/afhq/": "Images/Real/afhq.zip",
    "Real/celebahq/": "Images/Real/celebahq.zip",
    "GAN_based/": "Images/GAN_based.zip",
    "Other_based/": "Images/Other_based.zip",
    "Diffusion_based/ADM/": "Images/Diffusion_based/ADM.zip",
    "Diffusion_based/VQDM/": "Images/Diffusion_based/VQDM.zip",
    "Diffusion_based/Imagen/": "Images/Diffusion_based/Imagen.zip",
    "Diffusion_based/DDPM/": "Images/Diffusion_based/DDPM.zip",
    "Diffusion_based/DDIM/": "Images/Diffusion_based/DDIM.zip",
}

# Partial strata: one part of a multi-part tree, the smallest that still covers
# the stratum's share. SD and Midjourney each contribute BOTH tiers, so each
# needs its own archive -- the Advanced ones are what the tier weighting exists
# to buy, and taking only the cheap Typical parts would have made the corpus
# older than the benchmark rather than newer.
#
#   SD|1          Advanced/part_1   48.0 GB   need 16,591
#   Midjourney|1  Advanced/part_7   46.9 GB   need 19,218
#   SD|0          Typical/part_3    15.1 GB   need 12,359
#   Midjourney|0  Typical/part_4    32.8 GB   need  3,666  <- the poor-value one
PARTIAL = {
    "SD|1": "Images/Diffusion_based/SD/originalSD/Advanced/part_1.zip",
    "Midjourney|1": "Images/Diffusion_based/Midjourney/Advanced/part_7.zip",
    "SD|0": "Images/Diffusion_based/SD/originalSD/Typical/part_3.zip",
    "Midjourney|0": "Images/Diffusion_based/Midjourney/Typical/part_4.zip",
}

ROOT = Path(__file__).resolve().parents[2] / "data" / "wildfake_train"
IMAGES, ZIPS, AVAIL = ROOT / "images", ROOT / "_zips", ROOT / "available"
CONFIG = "configs/datasets/wildfake_train.yaml"


def download(member: str) -> Path:
    """One archive into _zips/, returning its local path."""
    from modelscope.hub.snapshot_download import dataset_snapshot_download

    local = ZIPS / Path(member).name
    if local.exists():
        print(f"  already downloaded: {local.name} ({local.stat().st_size / 2**30:.1f} GB)")
        return local

    ZIPS.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {member} ...", flush=True)
    got = dataset_snapshot_download(REPO, allow_file_pattern=[member], local_dir=str(ZIPS))
    src = Path(got) / member
    if not src.is_file():
        raise SystemExit(f"expected {src} after download; got {sorted(Path(got).rglob('*.zip'))}")
    src.replace(local)
    return local


def index(zip_path: Path, prefixes: set[str]) -> dict[str, str]:
    """Map metadata-relative path -> member name, for members we can place.

    The archives are not consistently rooted -- some carry `Real/x/...`, some
    only `x/...` -- so each member is matched by walking its own path down one
    component at a time until it starts with a prefix the metadata uses. That
    makes the mapping the archive's problem rather than a constant to maintain.
    """
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            parts = name.replace("\\", "/").split("/")
            for i in range(len(parts)):
                key = "/".join(parts[i:])
                if any(key.startswith(p) for p in prefixes):
                    out[key] = name
                    break
    return out


def plan(available: dict[str, Path]) -> set[str]:
    """The sample's chosen rows, as metadata-relative paths."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import copy

    from pipeline.config import load_dataset_config
    from pipeline.utils.imports import instantiate

    spec = copy.deepcopy(load_dataset_config(CONFIG).source)
    spec["args"]["available_lists"] = {g: str(p) for g, p in available.items()}
    src = instantiate(spec)

    chosen, _ = src._plan()
    return {
        row[src.path_column].strip().removeprefix("./")
        for i, row in src._reader()
        if i in chosen
    }


def extract(zip_path: Path, members: dict[str, str], wanted: set[str]) -> int:
    """Pull just the wanted members out, writing them at their metadata path."""
    todo = {k: m for k, m in members.items() if k in wanted}
    if not todo:
        return 0
    with zipfile.ZipFile(zip_path) as z:
        for i, (key, member) in enumerate(sorted(todo.items()), 1):
            dest = IMAGES / key
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as fsrc, open(dest, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            if i % 2000 == 0:
                print(f"    {i}/{len(todo)}", flush=True)
    return len(todo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="plan only, download nothing")
    p.add_argument("--resume", action="store_true", help="skip archives already extracted")
    p.add_argument("--keep-zips", action="store_true", help="do not delete after extracting")
    args = p.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)
    AVAIL.mkdir(parents=True, exist_ok=True)

    # -- phase 1: the two partial archives, and what they turn out to contain --
    available = {}
    for group, member in PARTIAL.items():
        # "|" is legal in a stratum key and illegal in a Windows filename.
        listing = AVAIL / f"{group.replace('|', '_')}.txt"
        if listing.exists():
            print(f"[{group}] availability list already written ({listing})")
        elif args.dry_run:
            print(f"[{group}] would download {member} and index it")
            continue
        else:
            print(f"[{group}] phase 1: {member}")
            z = download(member)
            found = index(z, {"Diffusion_based/"})
            listing.write_text("\n".join(sorted(found)), encoding="utf-8")
            print(f"  indexed {len(found)} members -> {listing.name}")
        available[group] = listing

    if args.dry_run and len(available) < len(PARTIAL):
        print("\n--dry-run: cannot plan until the partial archives are indexed.")
        return

    # ---------------------------- phase 2: the sample ------------------------
    print("\nphase 2: planning the sample")
    wanted = plan(available)
    print(f"  {len(wanted)} images to extract")
    if args.dry_run:
        return

    # ------------------------- phase 3: mine every archive -------------------
    archives = {**{v: k for k, v in COMPLETE.items()}}
    for group, member in PARTIAL.items():
        archives[member] = "Diffusion_based/"

    for n, (member, prefix) in enumerate(archives.items(), 1):
        head = f"[{n}/{len(archives)}] {Path(member).name}"
        needed = {w for w in wanted if w.startswith(prefix)} if prefix in COMPLETE else wanted
        if args.resume and needed and all((IMAGES / w).exists() for w in needed):
            print(f"{head}: all {len(needed)} present, skipping")
            continue
        print(f"{head}: {len(needed)} candidates")
        z = download(member)
        got = extract(z, index(z, {prefix.split('/')[0] + "/"}), wanted)
        print(f"  extracted {got}")
        if not args.keep_zips:
            z.unlink()
            print(f"  removed {z.name}")

    have = sum(1 for w in wanted if (IMAGES / w).exists())
    print(f"\ndone: {have}/{len(wanted)} images under {IMAGES}")
    if have < len(wanted):
        raise SystemExit("some images are missing -- rerun with --resume")


if __name__ == "__main__":
    main()
