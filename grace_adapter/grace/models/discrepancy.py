"""The discrepancy branch: read the drift instead of throwing it away.

RA-Det's finding is that generated images drift further in embedding space under
perturbation than real ones do. An adapter trained purely to *erase* drift is
therefore destroying forensic evidence while its reconstruction loss falls --
and doing so asymmetrically, which is worse than doing it uniformly.

The fix is to keep the quantity the adapter already computes:

    Δ = adapter(f_deg) − f_deg

Δ is the adapter's *estimate of the drift*, obtained without the clean image, as
a by-product of a module that was running anyway. RA-Det needs a second forward
pass on a deliberately perturbed image to get the same signal; here it is free.

It also breaks the restoration ceiling. A perfect restorer can at best recover
the clean-image score -- retention 1.0. A fused score that reads Δ can exceed it,
because the *magnitude of the damage* is information the clean image does not
contain.

This branch is the only part of GRACE that uses labels, which is why it is
trained in a separate stage against a frozen adapter (see
`grace.train.loop.train_discrepancy`). GRACE and GRACE-D therefore ship the same
adapter weights, bit for bit, and "the adapter is trained without labels" stays
literally true.
"""

import torch
import torch.nn as nn

from grace.splits.base import FeatureSpec


class DiscrepancyHead(nn.Module):
    """(Δ, ‖Δ‖, severity) -> one auxiliary logit.

    Three inputs, cheapest first:

    * **per-group norms of Δ.** For a `layers` split these are the per-block
      damage profile -- the same vector the per-layer gate produces, and the
      interpretability figure. For `vector` it is one number.
    * **a projection of Δ.** Direction, not just magnitude: which *way* the
      features moved is more informative than how far.
    * **predicted severity**, when available. Lets the head calibrate "is this
      drift large *for this much corruption*", which is the actually
      discriminative question -- a heavily degraded real image drifts a lot too.

    Norms are passed through `log1p`: drift magnitude spans orders of magnitude
    across levels, and an unsquashed input makes the first layer's job harder
    than it needs to be.
    """

    def __init__(
        self,
        spec: FeatureSpec,
        hidden: int = 256,
        proj: int = 64,
        use_severity: bool = True,
    ):
        super().__init__()
        self.spec = spec
        self.use_severity = use_severity

        self.proj = nn.Sequential(nn.LayerNorm(spec.dim), nn.Linear(spec.dim, proj))
        n_in = spec.n_groups + proj + (1 if use_severity else 0)
        self.net = nn.Sequential(
            nn.LayerNorm(n_in),
            nn.Linear(n_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def features(self, delta: torch.Tensor, severity: torch.Tensor | None = None) -> torch.Tensor:
        """The head's input vector, exposed so diagnostics can inspect it."""
        flat = delta if delta.ndim > 2 else delta.unsqueeze(1)   # (B, G, D)
        norms = torch.log1p(flat.norm(dim=-1))                   # (B, G)
        parts = [norms, self.proj(flat.mean(dim=1))]
        if self.use_severity:
            if severity is None:
                raise ValueError("head was built with use_severity=True but got severity=None")
            parts.append(severity.reshape(-1, 1))
        return torch.cat(parts, dim=-1)

    def forward(self, delta: torch.Tensor, severity: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(self.features(delta, severity)).squeeze(-1)


class FusedHead(nn.Module):
    """logit = logit_main + β · aux_logit,  β initialized to 0.

    The same defence the gate gives the adapter, applied to the fusion: at
    initialization GRACE-D is *exactly* GRACE, so any change in the reported
    numbers is attributable to what the auxiliary head learned rather than to
    having wired it in.
    """

    def __init__(self, aux: DiscrepancyHead):
        super().__init__()
        self.aux = aux
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        logit_main: torch.Tensor,
        delta: torch.Tensor,
        severity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return logit_main + self.beta * self.aux(delta, severity)
