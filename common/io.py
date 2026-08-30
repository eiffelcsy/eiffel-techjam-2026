"""Small IO helpers: json read/write, yaml read, path listing."""

import json
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def read_yaml(path: str | Path) -> Any:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def resolve_ref(entry: str | Path | dict) -> dict:
    """A config reference is either a path to a spec file or the same mapping inline."""
    return read_yaml(entry) if isinstance(entry, (str, Path)) else entry


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_fallback)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def list_images(root: str | Path, recursive: bool = True) -> list[Path]:
    """All decodable image files under root, sorted for stable ordering.

    Sorting is load-bearing: row order is the index that seeds every
    degradation, so an unsorted walk would change the eval set between runs.
    """
    root = Path(root)
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _fallback(o: Any) -> Any:
    """Let numpy scalars and Paths through json.dump."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
