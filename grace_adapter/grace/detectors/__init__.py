"""The adapted detectors, as the eval harness sees them.

`AdaptedDetector` is GRACE and GRACE-D: one input tensor, the harness's original
path. `FusedDetector` is GRACE-freq, and it is the only detector in this tree
that reads the image twice -- see its module docstring for why that cannot be
avoided downstream of preprocessing.
"""

from grace.detectors.adapted import AdaptedDetector
from grace.detectors.fused import FusedDetector

__all__ = ["AdaptedDetector", "FusedDetector"]
