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
    ):
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

    def _label_of(self, class_name: str) -> int | None:
        return _polarity(class_name, self.fake_classes, self.real_classes)

    def rows(self, out_dir: str | Path) -> Iterator[dict]:
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_id, self.config_name, split=self.split,
            revision=self.revision, streaming=self.streaming,
        )
        names = getattr(ds.features[self.label_column], "names", None)
        out_dir = Path(out_dir)

        kept = {REAL: 0, FAKE: 0}
        for i, ex in enumerate(ds):
            raw = ex[self.label_column]
            class_name = names[raw] if names is not None else str(raw)
            label = self._label_of(class_name)
            if label is None:
                continue
            if self.limit is not None and kept[label] >= self.limit:
                if all(v >= self.limit for v in kept.values()):
                    break
                continue

            path = out_dir / "images" / f"{i:08d}_{class_name}.png"
            _save(ex[self.image_column], path)
            kept[label] += 1
            yield {
                "path": str(path.resolve()),
                "label": label,
                "generator": self.generator if label == FAKE else "REAL",
                "split": self.split,
            }

        _check_split(kept[FAKE], kept[REAL], self.fake_classes)


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

        with open(self.csv_path, newline="") as f:
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
        from pipeline.utils.io import list_images

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
        from pipeline.utils.imports import instantiate

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
