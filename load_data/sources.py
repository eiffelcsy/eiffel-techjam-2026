"""Dataset sources: anything -> images on disk + manifest rows.

A source is the pipeline's only ingress point for data. It materializes some
external dataset into a directory of PNGs and yields the manifest rows that
describe them; everything downstream reads the manifest and nothing else. That
is what keeps the harness dataset-agnostic -- a new dataset is a new source
class plus a config file, never a change in here.

Label polarity is declared, never inferred. Datasets disagree on whether 0
means real or generated, and a silent flip produces a plausible-looking
(1 - AUC) that poisons every number downstream. Each source therefore takes the
set of class names that mean *generated* and maps everything else to real,
raising if that split turns out to be degenerate.
"""

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from PIL import Image

REAL, FAKE = 0, 1


@runtime_checkable
class Source(Protocol):
    """Materialize a dataset under out_dir, yielding manifest rows."""

    def rows(self, out_dir: str | Path) -> Iterator[dict]: ...


def _norm(rel: str) -> str:
    """A table's relative path, without the "./" many of them carry."""
    return str(rel).strip().removeprefix("./")


def _polarity(class_name: str, fake: set[str], real: set[str] | None) -> int | None:
    """FAKE, REAL, or None for a class this run does not evaluate.

    `real` unset means "everything not fake is real", which is right for a
    binary dataset and wrong for anything else -- naming both sets drops the
    classes in neither instead of quietly folding them into real.
    """
    key = class_name.lower()
    if key in fake:
        return FAKE
    if real is None:
        return REAL
    return REAL if key in real else None


def _check_split(n_fake: int, n_real: int, fake_classes) -> None:
    if n_fake == 0:
        raise ValueError(
            f"no example matched fake_classes={sorted(fake_classes)} -- every image "
            "would be labelled real. Check the class names against the dataset."
        )
    if n_real == 0:
        raise ValueError(
            f"every example matched fake_classes={sorted(fake_classes)} -- no real "
            "images left to score against."
        )


def _check_nonempty(n: int, what: str) -> None:
    """A fixed-label source needs at least one row, but not both classes.

    `_check_split` insists on both, which is right for an inferred-polarity
    dataset and wrong for a source that DECLARES every image one class -- an
    empty result there is still a bug worth failing on, but the failure is
    "nothing matched", not "you forgot the other class".
    """
    if n == 0:
        raise ValueError(
            f"the source produced no {what} rows -- every image would be dropped. "
            "Check the filters, skip/limit, or the dataset itself."
        )


def _save(img: Image.Image, path: Path) -> None:
    """Write pixels and nothing else.

    PIL's PNG writer falls back to `im.info["icc_profile"]`, so a plain save()
    copies the upstream container's colour profile into our PNG -- a per-source
    marker sitting in a file whose whole point is that its container says
    nothing about where it came from. Rebuilding from raw bytes leaves an empty
    info dict, so every materialized image carries the same (no) metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    Image.frombytes("RGB", rgb.size, rgb.tobytes()).save(path, "PNG")


class HFImageDatasetSource:
    """Any Hugging Face Hub image dataset with a label column.

    Parameters
    ----------
    dataset_id   : hub id, e.g. "some-org/some-dataset"
    split        : which split to pull; also recorded in the manifest's `split`
    fake_classes : class names meaning *generated*, matched case-insensitively
    real_classes : class names meaning *authentic*. Omit and everything not
                   fake counts as real; name them explicitly on a multi-class
                   dataset and anything in neither set is dropped rather than
                   folded into one side.
    generator    : what produced the fakes, recorded per fake row
    limit        : cap on images materialized per class, in dataset order
    streaming    : iterate the dataset over the network instead of downloading
                   it whole. Essential with `limit` on a large repo -- a
                   non-streaming load fetches every shard before the first row.
    fixed_label  : set (REAL/FAKE) to declare every row one class and skip the
                   label column -- for a single-class dataset with no label.
    skip         : skip the first N matched rows, so two sources over the same
                   dataset slice disjoint train/val ranges.
    manifest_split : the split recorded per row, decoupled from the HF `split`
                   (which names the dataset's own split, often just "train").
    resume       : skip the decode+re-encode for files already on disk from an
                   earlier run of the same source (same dataset, skip, limit and
                   out_dir). The row is still yielded.

    Class names come from the label column's ClassLabel names when it has them,
    and from `str(value)` when it is a plain integer column -- so an unnamed
    3-class label maps as fake_classes: ["1"], real_classes: ["0"].
    """

    def __init__(
        self,
        dataset_id: str,
        split: str,
        fake_classes: list[str],
        real_classes: list[str] | None = None,
        generator: str = "UNKNOWN",
        image_column: str = "image",
        label_column: str = "label",
        config_name: str | None = None,
        revision: str | None = None,
        limit: int | None = None,
        streaming: bool = False,
        fixed_label: int | None = None,
        skip: int = 0,
        manifest_split: str | None = None,
        resume: bool = False,
    ):
        if fixed_label is not None and fixed_label not in (REAL, FAKE):
            raise ValueError(f"fixed_label must be REAL or FAKE, not {fixed_label!r}")
        self.dataset_id = dataset_id
        self.split = split
        self.fake_classes = {str(c).lower() for c in fake_classes}
        self.real_classes = {str(c).lower() for c in real_classes} if real_classes else None
        self.generator = generator
        self.image_column = image_column
        self.label_column = label_column
        self.config_name = config_name
        self.revision = revision
        self.limit = limit
        self.streaming = streaming
        self.fixed_label = fixed_label
        self.skip = skip
        self.manifest_split = manifest_split
        self.resume = resume

    def _label_of(self, class_name: str) -> int | None:
        return _polarity(class_name, self.fake_classes, self.real_classes)

    def rows(self, out_dir: str | Path) -> Iterator[dict]:
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_id, self.config_name, split=self.split,
            revision=self.revision, streaming=self.streaming,
        )
        out_dir = Path(out_dir)
        names = (
            getattr(ds.features[self.label_column], "names", None)
            if self.fixed_label is None
            else None
        )

        kept = {REAL: 0, FAKE: 0}
        skipped = 0
        for i, ex in enumerate(ds):
            if self.fixed_label is not None:
                label = self.fixed_label
                class_name = "fake" if label == FAKE else "real"
            else:
                raw = ex[self.label_column]
                class_name = names[raw] if names is not None else str(raw)
                label = self._label_of(class_name)
                if label is None:
                    continue

            # `skip` counts matched rows, so two sources with the same dataset and
            # different `skip`/`limit` slice disjoint, ordered ranges -- how a
            # single split is cut into train and validation.
            if skipped < self.skip:
                skipped += 1
                continue
            if self.limit is not None and kept[label] >= self.limit:
                if self.fixed_label is not None or all(
                    v >= self.limit for v in kept.values()
                ):
                    break
                continue

            path = out_dir / "images" / f"{i:08d}_{class_name}.png"
            # `resume` reuses a file left by a previous (possibly interrupted)
            # run of the SAME source: same dataset, same skip/limit, same
            # out_dir, so `i` names the same image. The row is still counted and
            # yielded; only the decode+re-encode is skipped.
            if not (self.resume and path.exists()):
                _save(ex[self.image_column], path)
            kept[label] += 1
            yield {
                "path": str(path.resolve()),
                "label": label,
                "generator": self.generator if label == FAKE else "REAL",
                "split": self.manifest_split or self.split,
            }

        if self.fixed_label is None:
            _check_split(kept[FAKE], kept[REAL], self.fake_classes)
        else:
            _check_nonempty(
                kept[self.fixed_label],
                "generated" if self.fixed_label == FAKE else "real",
            )


class CsvMetadataSource:
    """A dataset that ships as a metadata CSV plus an unpacked tree of images.

    The shape of most benchmarks distributed as archives rather than through the
    Hub: label, provenance and official split live in a table, and the images
    sit wherever the archives were unpacked. WildFake is the worked example --
    its `test_metadata.csv` *is* the official test split, one row per image, and
    a subset of it is selected by path prefix.

    Unlike `HFImageDatasetSource` this **references images in place and copies
    nothing**, so `out_dir` is unused. That is not disk thrift. The Hub source is
    handed decoded PIL objects and has no choice but to re-encode them; these
    files are already the bytes the dataset authors shipped, and re-saving a JPEG
    as PNG would resample away exactly the compression artifacts an AIGC
    detector keys on -- turning a benchmark into a measurement of PIL.

    Parameters
    ----------
    csv_path     : the metadata table
    root         : directory the table's paths resolve against
    path_column  : column holding each image's path, relative to `root`
    label_column : column holding the real/generated label
    fake_values  : values of `label_column` meaning *generated*, matched
                   case-insensitively against `str(value)`
    real_values  : values meaning *authentic*. Omit and everything not fake
                   counts as real; name them on a table with a richer label and
                   anything in neither set is dropped rather than folded in.
    path_prefix  : keep only rows whose path starts with one of these, matched
                   after a leading "./" is stripped from both sides. This is the
                   subsetting knob: a dataset's own columns often do not
                   distinguish the part you want (WildFake files every COCO
                   image under one `coco` architecture, and only the path says
                   train2017 from val2017).
    where        : {column: value | [values]} equality filter, applied first
    generator    : what produced the fakes, recorded per fake row
    split        : recorded in the manifest's `split` column
    limit        : cap on rows kept per class, in table order
    on_missing   : "error" (default) or "skip" for a row whose file is absent.
                   Erroring is the default because the failure it catches is
                   silent otherwise: an archive unpacked one directory deeper
                   than expected yields a benchmark quietly missing most of its
                   images, and `_check_split` only notices if a whole class
                   vanishes. Use "skip" deliberately, when the table describes
                   more of the dataset than was downloaded.

    Table order is preserved: it becomes the manifest index, which is the image
    identity seeding every degradation.
    """

    def __init__(
        self,
        csv_path: str,
        root: str,
        fake_values: list,
        real_values: list | None = None,
        path_column: str = "path",
        label_column: str = "label",
        path_prefix: list[str] | str | None = None,
        where: dict | None = None,
        generator: str = "UNKNOWN",
        split: str = "test",
        limit: int | None = None,
        on_missing: str = "error",
    ):
        if on_missing not in ("error", "skip"):
            raise ValueError(f"on_missing must be 'error' or 'skip', not {on_missing!r}")
        self.csv_path = csv_path
        self.root = root
        self.fake_values = {str(v).lower() for v in fake_values}
        self.real_values = {str(v).lower() for v in real_values} if real_values else None
        self.path_column = path_column
        self.label_column = label_column
        prefixes = [path_prefix] if isinstance(path_prefix, str) else path_prefix
        self.path_prefix = [_norm(p) for p in prefixes] if prefixes else None
        self.where = {k: {str(x) for x in (v if isinstance(v, list) else [v])}
                      for k, v in (where or {}).items()}
        self.generator = generator
        self.split = split
        self.limit = limit
        self.on_missing = on_missing

    def _selects(self, row: dict) -> bool:
        for column, allowed in self.where.items():
            if str(row[column]) not in allowed:
                return False
        if self.path_prefix is None:
            return True
        rel = _norm(row[self.path_column])
        return any(rel.startswith(p) for p in self.path_prefix)

    def rows(self, out_dir: str | Path) -> Iterator[dict]:
        import csv

        root = Path(self.root).expanduser()
        kept = {REAL: 0, FAKE: 0}
        missing = 0

        # encoding is explicit: Python's default is the locale's, which is
        # cp1252 on a stock Windows box, and WildFake's tables carry CJK
        # filenames under Stable Diffusion's lora/ tree. Decoding those with the
        # locale codec raises part-way through the file, so the same config
        # would build on Linux and crash on Windows.
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for column in (self.path_column, self.label_column, *self.where):
                if column not in (reader.fieldnames or []):
                    raise KeyError(
                        f"{self.csv_path} has no column {column!r} -- "
                        f"it has {reader.fieldnames}"
                    )

            for row in reader:
                if not self._selects(row):
                    continue
                label = _polarity(str(row[self.label_column]), self.fake_values, self.real_values)
                if label is None:
                    continue
                if self.limit is not None and kept[label] >= self.limit:
                    if all(v >= self.limit for v in kept.values()):
                        break
                    continue

                path = root / _norm(row[self.path_column])
                if not path.is_file():
                    if self.on_missing == "error":
                        raise FileNotFoundError(
                            f"{path} is named by {self.csv_path} but is not on disk "
                            f"({kept[REAL] + kept[FAKE]} rows resolved before it). Check that "
                            f"root={self.root!r} is the directory the table's paths hang off, "
                            f"and that the archive holding this subset was unpacked. Set "
                            f"on_missing: skip only if the table deliberately describes more "
                            f"of the dataset than you downloaded."
                        )
                    missing += 1
                    continue

                kept[label] += 1
                yield {
                    "path": str(path.resolve()),
                    "label": label,
                    "generator": self.generator if label == FAKE else "REAL",
                    "split": self.split,
                }

        if missing:
            print(f"{type(self).__name__}: skipped {missing} row(s) with no file on disk")
        _check_split(kept[FAKE], kept[REAL], self.fake_values)


class StratifiedCsvSource:
    """A proportional sample of a metadata CSV, stratified by one column.

    `CsvMetadataSource` can subset a table but not *sample* it: its `limit` is
    the first N rows per class in table order, and WildFake's tables are sorted
    by generator, so `limit` there would take one contiguous block -- a few
    hundred consecutive BigGAN frames -- rather than a spread. This source
    answers the other question: "give me N rows total, with every architecture
    and every real corpus represented in the proportion it occupies in the
    corpus."

    Why not one `CsvMetadataSource` per group under a `ConcatSource`
    ------------------------------------------------------------------
    Because a per-architecture source is single-class by construction, and
    `_check_split` (rightly) raises on that: it cannot tell a deliberate
    fake-only shard from a mislabelled dataset. Stratification therefore has to
    happen inside one source that sees both classes at once.

    Held-out groups
    ---------------
    `exclude_groups` is the leakage guard, and it is checked rather than
    trusted: a name that matches zero rows raises. That matters because the
    names being excluded are usually the evaluation set's own -- a typo in
    "DALLE" would not fail, it would quietly train on the generator the
    benchmark reports, and every downstream number would be contaminated in a
    way no later assertion could see.

    Determinism
    -----------
    The sample is seeded per (seed, label, group) through blake2b rather than
    `hash()`, whose randomization across processes would give a different corpus
    on every run -- and a manifest whose row order seeds every degradation must
    be reproducible. Rows are emitted in table order, so the manifest index
    keeps meaning the same thing it does for every other source.

    Parameters
    ----------
    csv_paths        : one table, or several read in order as if concatenated
    root             : directory the tables' paths resolve against
    group_column     : the stratum. WildFake's `Architecture` names both the
                       generator of a fake and the source corpus of a real, so
                       one column strata-fies both classes.
    n_total          : rows in the finished manifest, real + fake
    real_fraction    : share of `n_total` that is real. None = the sampled
                       pool's own ratio. Set it to hold the ratio of a corpus
                       wider than the pool -- e.g. to keep WildFake's overall
                       real:fake balance while excluding one real corpus.
    exclude_groups   : `group_column` values to drop entirely, before sampling
    generator_column : record each fake's own group as its `generator`, instead
                       of one fixed string -- what makes a per-generator
                       breakdown possible later without rebuilding
    available_lists  : {group: path to a newline-separated file listing the
                       relative paths of that group that actually exist}. For a
                       stratum whose archive was downloaded only in part: the
                       group keeps its full proportional COUNT, but is drawn
                       from the listed subset instead of from the whole family.
                       Written by scripts/fetch_wildfake_train.py from each
                       archive's own zip index, so the sample and the extraction
                       are planned against exactly the same set of files. A
                       group whose list is shorter than its share raises, rather
                       than quietly under-filling the stratum.
    seed             : sampling seed; part of the manifest's identity

    Group allocation is proportional with largest-remainder rounding, so the
    parts sum to exactly `n_total`. A group smaller than its share contributes
    everything it has and the shortfall is redistributed over the groups with
    headroom left.
    """

    def __init__(
        self,
        csv_paths: list[str] | str,
        root: str,
        fake_values: list,
        real_values: list | None = None,
        path_column: str = "path",
        label_column: str = "label",
        group_column: str = "Architecture",
        n_total: int = 100_000,
        real_fraction: float | None = None,
        splits: list[dict] | None = None,
        exclude_groups: list[str] | None = None,
        tiers: list[dict] | None = None,
        where: dict | None = None,
        generator_column: str | None = None,
        generator: str = "UNKNOWN",
        available_lists: dict | None = None,
        split: str = "train",
        seed: int = 0,
        on_missing: str = "error",
    ):
        if on_missing not in ("error", "skip"):
            raise ValueError(f"on_missing must be 'error' or 'skip', not {on_missing!r}")
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, not {n_total}")
        if real_fraction is not None and not 0.0 < real_fraction < 1.0:
            raise ValueError(
                f"real_fraction must lie strictly between 0 and 1, not {real_fraction} "
                "-- a manifest with only one class cannot be scored."
            )
        self.csv_paths = [csv_paths] if isinstance(csv_paths, str) else list(csv_paths)
        self.root = root
        self.fake_values = {str(v).lower() for v in fake_values}
        self.real_values = {str(v).lower() for v in real_values} if real_values else None
        self.path_column = path_column
        self.label_column = label_column
        # One column, or several joined by "|". A composite key is how a tier
        # becomes a stratum in its own right: WildFake's `IsAdvanced` splits SD
        # and Midjourney into their current and previous generations, which
        # `Architecture` alone folds together and samples over blindly.
        self.group_column = (
            [group_column] if isinstance(group_column, str) else list(group_column)
        )
        # One manifest, several disjoint splits. Each names its own size and
        # class balance and is allocated over the SAME stratum sizes, so a
        # validation split can be balanced 50/50 while training keeps the
        # corpus's own ratio and both still carry every stratum in the same
        # proportion. Disjointness is by construction: a stratum draws all its
        # splits' rows in one sample and hands out consecutive slices, so no row
        # can reach two splits however the budgets are set.
        self.splits = [
            {
                "name": str(s.get("name", "train")),
                "n_total": int(s["n_total"]),
                "real_fraction": (
                    None if s.get("real_fraction") is None else float(s["real_fraction"])
                ),
            }
            for s in (splits or [{"name": split, "n_total": n_total,
                                  "real_fraction": real_fraction}])
        ]
        for s in self.splits:
            if s["n_total"] <= 0:
                raise ValueError(f"split {s['name']!r} has non-positive n_total")
            if s["real_fraction"] is not None and not 0.0 < s["real_fraction"] < 1.0:
                raise ValueError(
                    f"split {s['name']!r} has real_fraction {s['real_fraction']} -- it "
                    "must lie strictly between 0 and 1, or a split ends up single-class."
                )
        if len({s["name"] for s in self.splits}) != len(self.splits):
            raise ValueError("split names must be unique")

        self.n_total = sum(s["n_total"] for s in self.splits)
        self.real_fraction = real_fraction
        self.exclude_groups = {str(g) for g in (exclude_groups or [])}
        self.where = {k: {str(x) for x in (v if isinstance(v, list) else [v])}
                      for k, v in (where or {}).items()}

        # Ordered, first match wins. Each entry is {name, weight, where}; the
        # weights are relative and get normalized, so 50/35/15 and 10/7/3 mean
        # the same thing.
        self.tiers = []
        for spec in tiers or []:
            missing = {"name", "weight", "where"} - set(spec)
            if missing:
                raise ValueError(f"tier {spec.get('name', spec)!r} is missing {sorted(missing)}")
            if float(spec["weight"]) <= 0:
                raise ValueError(f"tier {spec['name']!r} has non-positive weight")
            self.tiers.append((
                str(spec["name"]),
                float(spec["weight"]),
                {k: {str(x) for x in (v if isinstance(v, list) else [v])}
                 for k, v in spec["where"].items()},
            ))
        if len({t[0] for t in self.tiers}) != len(self.tiers):
            raise ValueError("tier names must be unique")
        self.generator_column = generator_column
        self.generator = generator
        self.available_lists = dict(available_lists or {})
        self.split = split
        self.seed = seed
        self.on_missing = on_missing
        self._available: dict[str, set] | None = None

    def _selects(self, row: dict) -> bool:
        for column, allowed in self.where.items():
            if str(row[column]) not in allowed:
                return False
        return True

    def _group(self, row: dict) -> str:
        return "|".join(str(row[c]) for c in self.group_column)

    def _tier_of(self, row: dict) -> str | None:
        for name, _, cond in self.tiers:
            if all(str(row[c]) in allowed for c, allowed in cond.items()):
                return name
        return None

    def _excluded(self, group: str) -> str | None:
        """Which `exclude_groups` entry drops this stratum, if any.

        A composite key is matched either whole ("SD|0", one tier) or by its
        family ("DALLE", every tier of it). The leakage guards are family-level
        -- a generation of DALL-E is not safer to train on than another -- so
        naming the family has to keep working once tiers become strata.
        """
        if group in self.exclude_groups:
            return group
        family = group.split("|")[0]
        return family if family in self.exclude_groups else None

    def _reader(self):
        """Every table in order, as one stream of (global_index, row).

        The index counts *every* row read, filtered or not, so the counting pass
        and the emitting pass agree on what row 1_234_567 is.
        """
        import csv

        i = 0
        required = (
            self.path_column, self.label_column, *self.group_column, *self.where,
            *{c for _, _, cond in self.tiers for c in cond},
        )
        for csv_path in self.csv_paths:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for column in required:
                    if column not in (reader.fieldnames or []):
                        raise KeyError(
                            f"{csv_path} has no column {column!r} -- "
                            f"it has {reader.fieldnames}"
                        )
                for row in reader:
                    yield i, row
                    i += 1

    @staticmethod
    def _allocate(sizes: dict[str, int], budget: int) -> dict[str, int]:
        """Proportional shares summing to exactly `budget`, capped at group size."""
        if budget >= sum(sizes.values()):
            return dict(sizes)

        alloc = {g: 0 for g in sizes}
        room = {g: n for g, n in sizes.items() if n > 0}
        remaining = budget

        while remaining > 0 and room:
            total = sum(room.values())
            raw = {g: remaining * n / total for g, n in room.items()}
            take = {g: min(int(r), room[g]) for g, r in raw.items()}

            # Largest remainder, ties broken by name so the split is stable.
            short = remaining - sum(take.values())
            if short > 0:
                for g in sorted(room, key=lambda g: (-(raw[g] - int(raw[g])), g)):
                    if short == 0:
                        break
                    if take[g] < room[g]:
                        take[g] += 1
                        short -= 1
            if not any(take.values()):
                break

            for g, k in take.items():
                alloc[g] += k
                room[g] -= k
                remaining -= k
            room = {g: n for g, n in room.items() if n > 0}

        return alloc

    def _allocate_tiered(self, sizes, budget, group_tier, plan) -> dict[str, int]:
        """Split the budget across tiers by weight, then within a tier by size.

        The two levels answer different questions and must not be collapsed. The
        tier split is a JUDGEMENT -- how much of the corpus should be near the
        state of the art -- and is set by hand. The split inside a tier is a
        MEASUREMENT: once a tier's budget is fixed, its strata divide it in the
        proportion they occupy in WildFake, so nothing inside a tier is
        hand-weighted.
        """
        total_w = sum(w for _, w, _ in self.tiers)
        raw = {name: budget * w / total_w for name, w, _ in self.tiers}
        tier_budget = {name: int(v) for name, v in raw.items()}
        short = budget - sum(tier_budget.values())
        for name in sorted(raw, key=lambda n: (-(raw[n] - int(raw[n])), n)):
            if short == 0:
                break
            tier_budget[name] += 1
            short -= 1

        alloc: dict[str, int] = {}
        for name, weight, _ in self.tiers:
            members = {g: n for g, n in sizes.items() if group_tier.get(g) == name}
            if not members:
                raise ValueError(
                    f"tier {name!r} carries weight {weight} but no generated stratum "
                    "landed in it. Its budget cannot be spent, and the manifest would "
                    "come out short -- drop the tier or widen its `where`."
                )
            if sum(members.values()) < tier_budget[name]:
                raise ValueError(
                    f"tier {name!r} is allotted {tier_budget[name]} rows but its strata "
                    f"hold only {sum(members.values())}. Lower its weight or n_total."
                )
            got = self._allocate(members, tier_budget[name])
            alloc.update(got)
            plan["tiers"][name] = (tier_budget[name], sum(members.values()))
        return alloc

    def _plan(self) -> tuple[set[int], dict]:
        """Pass one: count the strata, then choose which rows survive."""
        import hashlib
        import random
        from array import array

        if self._available is None:
            self._available = {
                group: {
                    _norm(line)
                    for line in Path(listing).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
                for group, listing in self.available_lists.items()
            }

        pools: dict[tuple[int, str], array] = {}
        # Counted over EVERY row of the group, including ones no downloaded
        # archive holds. The distinction is load-bearing: a stratum's share must
        # come from how big it is in the corpus, not from how much of it was
        # downloaded, or fetching one part of Stable Diffusion would drop it
        # from 35% of the fake half to 2% and silently re-weight all the rest.
        true_sizes: dict[tuple[int, str], int] = {}
        group_tier: dict[str, str] = {}
        excluded_hits = {g: 0 for g in self.exclude_groups}

        for i, row in self._reader():
            group = self._group(row)
            hit = self._excluded(group)
            if hit is not None:
                excluded_hits[hit] += 1
                continue
            if not self._selects(row):
                continue
            label = _polarity(str(row[self.label_column]), self.fake_values, self.real_values)
            if label is None:
                continue
            if self.tiers and label == FAKE:
                tier = self._tier_of(row)
                if tier is None:
                    raise ValueError(
                        f"no tier matches the generated row {row[self.path_column]!r} "
                        f"(group {group!r}). Every generated stratum must land in a "
                        "tier, or its share of the corpus would be silently dropped "
                        "-- add a catch-all tier, or widen the last one's `where`."
                    )
                if group_tier.setdefault(group, tier) != tier:
                    raise ValueError(
                        f"stratum {group!r} spans tiers {group_tier[group]!r} and "
                        f"{tier!r}. Add the tier's distinguishing column to "
                        "group_column so the split is visible in the manifest."
                    )
            true_sizes[(label, group)] = true_sizes.get((label, group), 0) + 1
            if group in self._available:
                if _norm(row[self.path_column]) not in self._available[group]:
                    continue
            pools.setdefault((label, group), array("i")).append(i)

        missed = sorted(g for g, n in excluded_hits.items() if n == 0)
        if missed:
            raise ValueError(
                f"exclude_groups={missed} matched no row of "
                f"{'|'.join(self.group_column)!r}. "
                "These names are the held-out set's own, so a typo here would "
                "silently train on the data the benchmark reports -- fix the "
                "spelling rather than removing the entry."
            )
        if not true_sizes:
            raise ValueError(
                f"no row survived where={self.where} and "
                f"exclude_groups={sorted(self.exclude_groups)}"
            )

        sizes = {REAL: {}, FAKE: {}}
        for (label, group), n in true_sizes.items():
            sizes[label][group] = n
        for label, name in ((REAL, "real"), (FAKE, "generated")):
            if not sizes[label]:
                raise ValueError(
                    f"the sampling pool has no {name} rows left after "
                    f"exclude_groups={sorted(self.exclude_groups)}"
                )

        n_real_pool, n_fake_pool = sum(sizes[REAL].values()), sum(sizes[FAKE].values())
        plan: dict = {"splits": {}, "groups": {}, "group_tier": group_tier}

        # Allocate every split first, over the same stratum sizes, and only then
        # draw. Nothing is sampled until each stratum knows its whole demand, so
        # the splits can be cut from one sample and are disjoint by slicing
        # rather than by a subtraction that could quietly overlap.
        need: dict[tuple[int, str], list[tuple[str, int]]] = {}
        for spec in self.splits:
            name, n = spec["name"], spec["n_total"]
            fraction = (
                spec["real_fraction"]
                if spec["real_fraction"] is not None
                else n_real_pool / (n_real_pool + n_fake_pool)
            )
            budgets = {REAL: round(n * fraction)}
            budgets[FAKE] = n - budgets[REAL]
            entry = {"fraction": fraction, "n_total": n, "tiers": {}, "groups": {}}

            for label in (REAL, FAKE):
                if self.tiers and label == FAKE:
                    alloc = self._allocate_tiered(sizes[FAKE], budgets[FAKE], group_tier, entry)
                else:
                    alloc = self._allocate(sizes[label], budgets[label])
                for group, k in alloc.items():
                    need.setdefault((label, group), []).append((name, k))
                    entry["groups"][(label, group)] = k
            plan["splits"][name] = entry

        assignment: dict[int, str] = {}
        for (label, group), parts in sorted(need.items()):
            total = sum(k for _, k in parts)
            idx = pools.get((label, group), array("i"))
            if total > len(idx):
                detail = " + ".join(f"{k} for {n}" for n, k in parts)
                raise ValueError(
                    f"{group!r} needs {total} rows ({detail}) for its "
                    f"{sizes[label][group]}-row share of the corpus, but only "
                    f"{len(idx)} are available. "
                    + (
                        f"The archive listed in available_lists[{group!r}] is too small "
                        "-- fetch another part, or lower n_total."
                        if group in self._available
                        else "This should not be reachable; the pool is the group."
                    )
                )
            material = f"{self.seed}:{label}:{group}".encode()
            digest = hashlib.blake2b(material, digest_size=8).digest()
            rng = random.Random(int.from_bytes(digest, "big"))
            drawn = rng.sample(idx, total) if total < len(idx) else list(idx)

            at = 0
            for name, k in parts:
                for i in drawn[at:at + k]:
                    assignment[i] = name
                at += k
            plan["groups"][(label, group)] = (sizes[label][group], total)

        return assignment, plan

    def rows(self, out_dir: str | Path) -> Iterator[dict]:
        assignment, plan = self._plan()

        print(
            f"{type(self).__name__}: sampling {self.n_total} of "
            f"{sum(n for n, _ in plan['groups'].values())} eligible rows "
            f"(seed {self.seed})"
        )
        for name, entry in plan["splits"].items():
            print(
                f"  [{name}] {entry['n_total']} rows, "
                f"real fraction {entry['fraction']:.4f}"
                + (
                    "  tiers: " + ", ".join(f"{t}={b}" for t, (b, _) in entry["tiers"].items())
                    if entry["tiers"]
                    else ""
                )
            )
        for (label, group), (have, take) in sorted(
            plan["groups"].items(), key=lambda kv: (kv[0][0], -kv[1][1])
        ):
            kind = "real" if label == REAL else "fake"
            per = "/".join(
                str(e["groups"].get((label, group), 0)) for e in plan["splits"].values()
            )
            print(f"  {kind:4s} {group:<16s} {take:>6d} of {have:>7d}   ({per})")

        root = Path(self.root).expanduser()
        kept = {REAL: 0, FAKE: 0}
        per_split = {s["name"]: {REAL: 0, FAKE: 0} for s in self.splits}
        missing: dict[str, int] = {}

        for i, row in self._reader():
            if i not in assignment:
                continue
            label = _polarity(str(row[self.label_column]), self.fake_values, self.real_values)
            group = self._group(row)

            path = root / _norm(row[self.path_column])
            if not path.is_file():
                if self.on_missing == "error":
                    raise FileNotFoundError(
                        f"{path} is named by the metadata table but is not on disk "
                        f"({kept[REAL] + kept[FAKE]} rows resolved before it). The "
                        f"archive holding {group!r} is probably not unpacked under "
                        f"root={self.root!r}. Do NOT set on_missing: skip to get past "
                        f"this -- the sample is proportional, and silently dropping "
                        f"one archive re-weights every stratum in the manifest."
                    )
                missing[group] = missing.get(group, 0) + 1
                continue

            kept[label] += 1
            per_split[assignment[i]][label] += 1
            yield {
                "path": str(path.resolve()),
                "label": label,
                "generator": (
                    (row[self.generator_column] if self.generator_column else self.generator)
                    if label == FAKE
                    else "REAL"
                ),
                "split": assignment[i],
            }

        if missing:
            total = sum(missing.values())
            detail = ", ".join(f"{g}:{n}" for g, n in sorted(missing.items()))
            print(
                f"{type(self).__name__}: WARNING -- skipped {total} row(s) with no file "
                f"on disk ({detail}). The manifest is NO LONGER proportional to the "
                f"corpus; the groups above are under-represented by exactly these counts."
            )
        # Per split, not just overall: a validation split with no generated rows
        # cannot be scored, and an aggregate check would pass it as long as the
        # training split had some.
        for name, n in per_split.items():
            if not n[FAKE] or not n[REAL]:
                raise ValueError(
                    f"split {name!r} came out {n[REAL]} real / {n[FAKE]} generated -- "
                    "a single-class split cannot be scored. Check its real_fraction "
                    "and n_total."
                )
        _check_split(kept[FAKE], kept[REAL], self.fake_values)


class ImageDirSource:
    """Local directories of images, labelled by directory name.

    `fake_dirs` names the directory components that mean *generated* -- matched
    case-insensitively against every part of each image's path.
    """

    def __init__(
        self,
        roots: list[str],
        fake_dirs: list[str],
        generator: str = "UNKNOWN",
        split: str = "val",
    ):
        self.roots = roots
        self.fake_dirs = {d.lower() for d in fake_dirs}
        self.generator = generator
        self.split = split

    def rows(self, out_dir: str | Path) -> Iterator[dict]:
        from common.io import list_images

        n = {REAL: 0, FAKE: 0}
        for root in self.roots:
            for path in list_images(root):
                parts = {p.lower() for p in path.parts}
                label = FAKE if parts & self.fake_dirs else REAL
                n[label] += 1
                yield {
                    "path": str(path.resolve()),
                    "label": label,
                    "generator": self.generator if label == FAKE else "REAL",
                    "split": self.split,
                }

        _check_split(n[FAKE], n[REAL], self.fake_dirs)


class SampledCsvSingleClassSource:
    """A deterministic sample of ONE class from a metadata CSV, disjoint from a
    set of already-used paths, split into train and validation.

    For adding MORE of a class to an existing manifest without touching the
    manifest's own rows: the existing rows' paths are passed in and skipped, and
    the remainder is sampled deterministically per group so the groups keep
    their corpus proportions. WildFake reals are the worked example -- laion5b
    and imagenet are the two groups, `label_values: ["0"]` selects them, and the
    existing manifest's real paths are the exclusion set.

    Unlike `StratifiedCsvSource` this emits one class only, so it skips the
    two-class `_check_split` and the tier machinery -- and because it is a
    *complement* to an existing sample, it takes an explicit exclusion set
    rather than trusting a seed never to collide.

    Parameters
    ----------
    csv_paths     : one table, or several read in order as if concatenated
    root          : directory the tables' paths resolve against
    label_values  : values of `label_column` meaning the class to sample
    group_column  : the stratum (WildFake's `Architecture`), split proportionally
    where         : {column: [values]} equality filter, applied before sampling
    n_train / n_val : rows to emit per split; both may be zero, not negative
    exclude_paths : normalized relative paths already used elsewhere, skipped
    label         : manifest label emitted for every row (REAL or FAKE)
    generator     : manifest generator for every emitted row
    seed          : part of the sample's identity
    on_missing    : "error" or "skip" for a row whose file is absent
    """

    def __init__(
        self,
        csv_paths,
        root,
        label_values,
        group_column: str = "Architecture",
        path_column: str = "Image_path",
        label_column: str = "IsFake",
        where: dict | None = None,
        n_train: int = 0,
        n_val: int = 0,
        exclude_paths: list[str] | None = None,
        label: int = REAL,
        generator: str = "REAL",
        seed: int = 0,
        on_missing: str = "error",
    ):
        if on_missing not in ("error", "skip"):
            raise ValueError(f"on_missing must be 'error' or 'skip', not {on_missing!r}")
        if label not in (REAL, FAKE):
            raise ValueError(f"label must be REAL or FAKE, not {label!r}")
        if n_train < 0 or n_val < 0 or (n_train + n_val) <= 0:
            raise ValueError("n_train + n_val must be positive")
        self.csv_paths = [csv_paths] if isinstance(csv_paths, str) else list(csv_paths)
        self.root = root
        self.label_values = {str(v) for v in label_values}
        self.group_column = group_column
        self.path_column = path_column
        self.label_column = label_column
        self.where = {k: {str(x) for x in (v if isinstance(v, list) else [v])}
                      for k, v in (where or {}).items()}
        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.exclude = {_norm(p) for p in (exclude_paths or [])}
        self.label = label
        self.generator = generator
        self.seed = seed
        self.on_missing = on_missing

    def _selects(self, row: dict) -> bool:
        for column, allowed in self.where.items():
            if str(row[column]) not in allowed:
                return False
        return True

    def _reader(self):
        import csv

        i = 0
        required = {self.path_column, self.label_column, self.group_column, *self.where}
        for csv_path in self.csv_paths:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for column in required:
                    if column not in (reader.fieldnames or []):
                        raise KeyError(
                            f"{csv_path} has no column {column!r} -- it has {reader.fieldnames}"
                        )
                for row in reader:
                    yield i, row
                    i += 1

    def _plan(self) -> dict[int, str]:
        """Choose which rows survive, as `{global_index: "train" | "validation"}`.

        Split from `rows()` so a fetch script can learn the sample before the
        files exist: the chosen rows' relative paths are the download list. The
        same `_reader()` order is used by both, so an index here means the same
        row there.
        """
        import hashlib
        import random

        pools: dict[str, list[int]] = {}
        sizes: dict[str, int] = {}
        for i, row in self._reader():
            if str(row[self.label_column]) not in self.label_values:
                continue
            if not self._selects(row):
                continue
            group = str(row[self.group_column])
            sizes[group] = sizes.get(group, 0) + 1
            if _norm(row[self.path_column]) in self.exclude:
                continue
            pools.setdefault(group, []).append(i)

        if not sizes:
            raise ValueError(
                f"no row matched label_values={sorted(self.label_values)} and "
                f"where={self.where or {}}"
            )

        train_alloc = StratifiedCsvSource._allocate(sizes, self.n_train)
        val_alloc = StratifiedCsvSource._allocate(sizes, self.n_val)

        assignment: dict[int, str] = {}
        for group in sorted(set(train_alloc) | set(val_alloc)):
            t, v = train_alloc.get(group, 0), val_alloc.get(group, 0)
            idx = pools.get(group, [])
            if t + v > len(idx):
                raise ValueError(
                    f"{group!r} needs {t + v} rows ({t} train + {v} val) but only "
                    f"{len(idx)} are available after exclusion. Lower n_train/n_val "
                    f"or widen `where`."
                )
            rng = random.Random(
                int.from_bytes(
                    hashlib.blake2b(f"{self.seed}:{group}".encode(), digest_size=8).digest(),
                    "big",
                )
            )
            drawn = rng.sample(idx, t + v) if t + v < len(idx) else list(idx)
            for k, i in enumerate(drawn):
                assignment[i] = "train" if k < t else "validation"
        return assignment

    def rows(self, out_dir: str | Path) -> Iterator[dict]:
        assignment = self._plan()
        root = Path(self.root).expanduser()
        n = 0
        missing: dict[str, int] = {}
        for i, row in self._reader():
            if i not in assignment:
                continue
            path = root / _norm(row[self.path_column])
            if not path.is_file():
                if self.on_missing == "error":
                    raise FileNotFoundError(
                        f"{path} is named by a metadata table but is not on disk "
                        f"({n} rows resolved before it)."
                    )
                missing[str(row[self.group_column])] = (
                    missing.get(str(row[self.group_column]), 0) + 1
                )
                continue
            n += 1
            yield {
                "path": str(path.resolve()),
                "label": self.label,
                "generator": self.generator,
                "split": assignment[i],
            }

        if missing:
            total = sum(missing.values())
            detail = ", ".join(f"{g}:{c}" for g, c in sorted(missing.items()))
            print(
                f"{type(self).__name__}: WARNING -- skipped {total} row(s) with no file "
                f"on disk ({detail})."
            )
        _check_nonempty(n, "real" if self.label == REAL else "generated")


class ConcatSource:
    """Several sources into one manifest -- the way a dataset gets a train split.

    Every other source materializes exactly one split, which is all evaluation
    ever needs: the harness scores a held-out set and nothing else. Training a
    classifier head against the same dataset (`grace_adapter`'s stage 0) needs
    two, drawn from the same distribution, disjoint by construction, and
    described in one place so that "the probe was fit on these rows and selected
    on those" is a property of the manifest rather than of a shell command.

    Parameters
    ----------
    sources : `{target, args}` specs, built in order and chained
    prefix  : if true (default), each source writes under its own subdirectory
              of `out_dir`

    `prefix` is not a convenience. `HFImageDatasetSource` names files by their
    position in the split it is iterating, so two splits of the same dataset both
    start at `images/00000000_0.png` and the second silently overwrites the
    first -- producing a manifest whose train and validation rows point at the
    same pixels. Subdirectories are named from each source's `split` attribute,
    falling back to its position.
    """

    def __init__(self, sources: list[dict], prefix: bool = True):
        from common.imports import instantiate

        if not sources:
            raise ValueError("ConcatSource needs at least one source")
        self.sources = [instantiate(s) for s in sources]
        self.prefix = prefix

    def _dir_for(self, i: int, source) -> str:
        name = str(getattr(source, "split", "") or f"part{i}")
        return name.replace("/", "_")

    def rows(self, out_dir: str | Path) -> Iterator[dict]:
        out_dir = Path(out_dir)
        # Label polarity is not re-checked here: each child runs `_check_split`
        # over its own rows at the end of its generator, which is the stricter
        # test -- an aggregate check would pass a validation split with no fakes
        # in it as long as the training split had some.
        seen: set[str] = set()
        for i, source in enumerate(self.sources):
            target = out_dir / self._dir_for(i, source) if self.prefix else out_dir
            for row in source.rows(target):
                if row["path"] in seen:
                    raise ValueError(
                        f"{row['path']} was produced by two sources. Set prefix: true "
                        f"(the default) so each writes into its own subdirectory -- "
                        f"otherwise the later source overwrites the earlier one's "
                        f"images and the manifest points at the wrong pixels."
                    )
                seen.add(row["path"])
                yield row
