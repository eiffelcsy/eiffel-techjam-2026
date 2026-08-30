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
#
# SCOPED DOWN on 2026-08-30, from a 120k corpus to 60k, because the full fetch
# was ~173 GiB of remaining download at a measured 1.16 MB/s -- about 42 hours,
# before any of it could be cached. What was dropped and why:
#
#   ADM, VQDM, Imagen, DDPM, DDIM   62.6 GiB   all five serve `typical_diffusion`,
#                                              the same tier SD Typical already
#                                              covers with 38,897 members on disk
#                                              against a need of ~14,283. Pure
#                                              within-tier generator diversity.
#   Other_based                     12.4 GiB   MAGE/VQVAE/VQGAN/MAE are
#                                              autoencoders, and GAN_based alone
#                                              fills the legacy tier's ~6,121.
#   church, ffhq, afhq, celebahq     2.6 GiB   single-subject corpora (churches,
#                                              faces, animals). Dropping them
#                                              REDUCES the D1 content confound:
#                                              the reals left are broad-content,
#                                              like the fakes they are scored
#                                              against.
#   Midjourney Typical part_4       27.0 GiB   see PARTIAL below.
#
# `GAN_based` is kept at 44.1 GiB despite being the single largest remaining
# item, and despite supplying only ~6,121 images. GAN upsampling leaves
# checkerboard artefacts in the high frequency band, which makes it the family
# most likely to show a clean signature to the frequency branch this corpus is
# being built to test. Dropping it would remove the best-case evidence.
#
# `laion5b` is kept at 23.1 GiB for the same kind of reason: it is the only
# broad-content real corpus at scale, and without it the real half is faces and
# churches while the fake half is diverse generated scenes -- a content shortcut
# of exactly the kind the 200x200 finding already cost this project.
COMPLETE = {
    "Real/laion5b/": "Images/Real/laion5b.zip",
    "Real/imagenet/": "Images/Real/imagenet.zip",
    "GAN_based/": "Images/GAN_based.zip",
}

# Partial strata: one part of a multi-part tree, the smallest that still covers
# the stratum's share. SD and Midjourney each contribute BOTH tiers, so each
# needs its own archive -- the Advanced ones are what the tier weighting exists
# to buy, and taking only the cheap Typical parts would have made the corpus
# older than the benchmark rather than newer.
#
#   SD|1          Advanced/part_1   48.0 GB   need  9,448   ON DISK
#   Midjourney|1  Advanced/part_7   46.9 GB   need 10,957   ON DISK
#   SD|0          Typical/part_3    15.1 GB   need 14,283   ON DISK
#
# Needs are train + validation combined -- the manifest carries both splits and
# every image of both has to be on disk. All three are already downloaded, so
# the fakes cost nothing further: part_1 holds 32,503 members, part_7 31,996 and
# part_3 38,897, each comfortably above its need.
#
# `Midjourney|0` (Typical/part_4, 32.8 GB) was dropped with ~27 GB still to
# fetch. SD Typical alone fills the typical tier, so the only thing part_4 buys
# is a second generator inside it -- while `advanced_diffusion`, the tier that
# carries 50% of the fake weight, keeps both SD and Midjourney either way. The
# stratum is excluded by whole key in wildfake_train.yaml, so its rows do not
# count toward the tier's size and SD|0 takes the whole share.
PARTIAL = {
    "SD|1": "Images/Diffusion_based/SD/originalSD/Advanced/part_1.zip",
    "Midjourney|1": "Images/Diffusion_based/Midjourney/Advanced/part_7.zip",
    "SD|0": "Images/Diffusion_based/SD/originalSD/Typical/part_3.zip",
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


def index(zip_path: Path, prefixes: set[str], tails: dict | None = None) -> dict[str, str]:
    """Map metadata-relative path -> member name, for members we can place.

    The archives are not consistently rooted, in BOTH directions, and the two
    cases need different treatment:

      rooted HIGHER   member `Images/Real/laion5b/x.png` for metadata
                      `Real/laion5b/x.png`. Walk the member's own path down one
                      component at a time until it starts with a metadata prefix.

      rooted DEEPER   member `silk_scarf/5952.png` for metadata
                      `Diffusion_based/SD/originalSD/Typical/silk_scarf/5952.png`.
                      No suffix of the member can start with `Diffusion_based/`,
                      because the archive has already stripped it. The missing
                      components have to come from the metadata instead.

    Only the first case was handled, and every partial archive is the second, so
    all three availability lists came out empty -- which would have let phase 2
    plan a sample from four strata it believed held nothing. `tails` is the fix:
    a {last-component: [metadata paths]} index built from the CSV, against which
    a deep-rooted member is matched by its own tail.

    Matching is by the last one or two components. Bare basenames are unique for
    the hash-named archives (Midjourney's members are flat md5 filenames) but
    emphatically not for the numeric ones -- `0.png` recurs in every category
    directory of SD Typical -- so a one-component match is accepted only when it
    is unambiguous, and two components disambiguate the rest.
    """
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            parts = name.replace("\\", "/").split("/")
            placed = False
            for i in range(len(parts)):
                key = "/".join(parts[i:])
                if any(key.startswith(p) for p in prefixes):
                    out[key] = name
                    placed = True
                    break
            if placed or not tails:
                continue
            # Deep-rooted: ask the metadata where this member belongs.
            for depth in (2, 1):
                if len(parts) < depth:
                    continue
                candidates = tails.get("/".join(parts[-depth:]))
                if candidates and len(candidates) == 1:
                    out[candidates[0]] = name
                    break
    return out


def metadata_tails(prefix: str) -> dict[str, list]:
    """{last-one-or-two path components: [metadata paths]} under `prefix`.

    Built once per archive family and handed to `index` so a deep-rooted member
    can be placed at the path the manifest will look for it at. Entries with
    more than one candidate are kept rather than dropped, so `index` can see the
    ambiguity and decline instead of guessing wrong.
    """
    import csv

    tails: dict[str, list] = {}
    for csv_path in _metadata_csvs():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = row["Image_path"].strip().removeprefix("./")
                if not path.startswith(prefix):
                    continue
                parts = path.split("/")
                for depth in (1, 2):
                    if len(parts) >= depth:
                        tails.setdefault("/".join(parts[-depth:]), []).append(path)
    return tails


def _metadata_csvs() -> list[Path]:
    base = ROOT / "split_train_test" / "csv_file" / "total_split"
    return [base / "test_metadata.csv", base / "train_metadata.csv"]


def plan(available: dict[str, Path]) -> set[str]:
    """The sample's chosen rows, as metadata-relative paths."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import copy

    from pipeline.config import load_dataset_config
    from common.imports import instantiate

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
        # An EMPTY list is treated as not written. It is what a failed index
        # leaves behind, and the previous `listing.exists()` guard made that
        # failure sticky: three 0-byte lists, and every rerun skipping straight
        # past them into a phase-2 plan that believed four strata held nothing.
        if listing.exists() and listing.stat().st_size > 0:
            print(f"[{group}] availability list already written ({listing})")
        elif args.dry_run:
            print(f"[{group}] would download {member} and index it")
            continue
        else:
            print(f"[{group}] phase 1: {member}")
            z = download(member)
            found = index(z, {"Diffusion_based/"}, metadata_tails("Diffusion_based/"))
            if not found:
                raise SystemExit(
                    f"indexed 0 members of {z.name}. Its members could not be "
                    f"placed at any metadata path -- check the archive's rooting "
                    f"against Image_path in the metadata CSVs before rerunning."
                )
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
        family = prefix.split("/")[0] + "/"
        got = extract(z, index(z, {family}, metadata_tails(family)), wanted)
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
