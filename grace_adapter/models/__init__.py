"""The adapter, the auxiliary heads, and the factory that shapes them.

The frequency branch's enricher is a sibling of this package, not a member of
it -- see `freq_branch.models`.
"""

from grace_adapter.models.adapter import GatedResidualAdapter
from grace_adapter.models.discrepancy import DiscrepancyHead, FusedHead
from grace_adapter.models.factory import (
    build_adapter, build_discrepancy_head, build_severity_head, load_adapter,
    save_adapter,
)
from grace_adapter.models.severity import SeverityHead

__all__ = [
    "DiscrepancyHead", "FusedHead", "GatedResidualAdapter", "SeverityHead",
    "build_adapter", "build_discrepancy_head", "build_severity_head",
    "load_adapter", "save_adapter",
]
