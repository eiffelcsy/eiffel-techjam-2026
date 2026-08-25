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
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "PNG")


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
        """FAKE, REAL, or None for a class this run does not evaluate."""
        key = class_name.lower()
        if key in self.fake_classes:
            return FAKE
        if self.real_classes is None:
            return REAL
        return REAL if key in self.real_classes else None

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
