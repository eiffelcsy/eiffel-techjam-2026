"""The frequency branch's detector, as the eval harness sees it.

`FusedDetector` is GRACE-freq, and it is the only detector in this tree that
reads the image twice -- see its module docstring for why that cannot be
avoided downstream of preprocessing.
"""

from freq_branch.detectors.fused import FusedDetector

__all__ = ["FusedDetector"]
