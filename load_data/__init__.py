"""Raw dataset sourcing and manifest building.

    load_data/sources.py    how to pull rows from the Hub, a CSV+images tree, ...
    load_data/manifest.py   the manifest.parquet every other stage reads
    load_data/config.py     DatasetConfig -- what a dataset IS
"""

from load_data.config import DatasetConfig, load_dataset_config
from load_data.manifest import build_manifest, load_manifest, sample_eval_subset
from load_data.sources import HFImageDatasetSource, ImageDirSource, Source

__all__ = [
    "DatasetConfig", "load_dataset_config",
    "build_manifest", "load_manifest", "sample_eval_subset",
    "HFImageDatasetSource", "ImageDirSource", "Source",
]
