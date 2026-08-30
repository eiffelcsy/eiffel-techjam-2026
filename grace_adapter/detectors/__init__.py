"""The GRACE branch's detector, as the eval harness sees it.

`AdaptedDetector` is GRACE and GRACE-D: one input tensor, the harness's
original path. The frequency branch's `FusedDetector` is a sibling of this
package, not a member of it -- see `freq_branch.detectors`.
"""

from grace_adapter.detectors.adapted import AdaptedDetector

__all__ = ["AdaptedDetector"]
