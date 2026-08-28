"""The adapter. One class, because layout is a gate shape and not a hierarchy.

    y = f + g ⊙ MLP(LN(f)),    g = sigmoid(gate_logit)

Everything operates on the last axis, so a `(B, D)` vector, a `(B, T, D)` token
stack and a `(B, L, D)` layer stack all run through the same code with the same
weights shared across the group axis. The only thing that differs is the shape
of the gate:

    vector  gate_shape = (D,)       per-channel
    tokens  gate_shape = (D,)       shared across tokens -- corruption is a
                                    property of the image, not of token position
    layers  gate_shape = (L, D)     one gate vector per layer, MLP still shared

`gate_shape` is chosen by `grace.models.factory.build_adapter`, so nothing else
in the project branches on layout. The `(L, D)` gate is also the interpretability
output: mean it over `D` and you have "how much correction does each encoder
block need", per degradation.

Four requirements, in priority order:

1. **Identity at initialization**, exactly. The last projection of every block is
   zero-initialized, so the adapter returns its input bit-for-bit whatever the
   gate, the noise, or the severity conditioning happen to be. Without this a
   clean-AUC change is unattributable: did the adapter fail, or merely perturb?
2. **Gated residual.** The adapter proposes a correction; a learned gate decides
   how much to apply. Ungated, it over-corrects clean inputs -- the failure mode
   that trades away the number people actually quote.
3. **Layout-agnostic**, as above.

Two optional inputs, both no-ops when absent and both identity-preserving at
init:

* **`z` -- noise, for posterior sampling.** `E[h(f)] != h(E[f])` for any
  nonlinear head, so averaging the *logits* of k sampled corrections is not the
  same as one deterministic pass. k adapter passes cost microseconds against one
  trunk pass. NOTE: under point-wise reconstruction losses alone the optimal
  policy is to ignore `z` -- posterior collapse. Noise earns its keep only
  alongside the sliced-Wasserstein term, which rewards matching the *spread* of
  the clean distribution that a conditional mean under-disperses. See
  `grace.train.losses.sliced_wasserstein`.
* **`severity` -- FiLM on the gate logit.** One mildly JPEG-ed image and one
  six-operation wreck need different correction magnitudes. Zero-initialized, so
  conditioning starts as a no-op. This scalar is the thin conditioning that
  `grace.models.prompts` (FUTURE) replaces with a soft mixture over learnable
  degradation prompts; keeping it behind one call site makes that a drop-in.

Log `gate().mean()` every step. It should climb off 0.018 and plateau around
0.1-0.5. Saturating at 1.0 is over-correction; sitting at init means the
alignment term is too weak against the identity term.
"""

import torch
import torch.nn as nn

GATE_INIT = -4.0
"""sigmoid(-4) ~= 0.018. Belt to the zero-init's braces: the adapter starts as a
near-no-op even if a future change breaks the exact-identity guarantee."""


def _expand(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Insert singleton axes after the batch axis so `t` broadcasts against `ref`.

    Needed only for batched conditioners: a `(B, D)` gate against a `(B, T, D)`
    feature would otherwise try to align `B` with `T`. Unbatched `(D,)` and
    `(L, D)` gates already broadcast correctly and are passed through.
    """
    while t.ndim < ref.ndim:
        t = t.unsqueeze(1)
    return t


class GatedResidualAdapter(nn.Module):
    """f_degraded -> f_clean_estimate, over the last dimension only.

    Parameters
    ----------
    dim         : channel width, `FeatureSpec.dim`
    gate_shape  : () for a scalar gate, (dim,) per-channel, (n_groups, dim)
                  per-group. Defaults to per-channel.
    bottleneck  : hidden width of each residual block
    n_blocks    : number of residual blocks
    dropout     : on the bottleneck activation
    noise_dim   : 0 disables posterior sampling entirely
    severity_film : enable FiLM conditioning of the gate on a scalar severity
    """

    def __init__(
        self,
        dim: int,
        gate_shape: tuple[int, ...] | None = None,
        bottleneck: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.0,
        noise_dim: int = 0,
        severity_film: bool = False,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        self.dim = dim
        self.noise_dim = noise_dim
        self.gate_shape = (dim,) if gate_shape is None else tuple(gate_shape)

        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_blocks))
        self.fc1 = nn.ModuleList(nn.Linear(dim, bottleneck) for _ in range(n_blocks))
        self.fc2 = nn.ModuleList(nn.Linear(bottleneck, dim) for _ in range(n_blocks))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Zero-init the *last* projection of each block: the correction is
        # identically zero at t=0 regardless of gate, noise or severity, so
        # `test_identity_at_init` passes exactly rather than approximately.
        for layer in self.fc2:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.gate_logit = nn.Parameter(torch.full(self.gate_shape, GATE_INIT))

        # Noise enters the bottleneck, not the residual stream: it perturbs the
        # proposed correction rather than the feature being corrected.
        self.noise = (
            nn.ModuleList(nn.Linear(noise_dim, bottleneck, bias=False) for _ in range(n_blocks))
            if noise_dim > 0
            else None
        )

        self.film = nn.Linear(1, 2 * dim) if severity_film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    @property
    def stochastic(self) -> bool:
        return self.noise is not None

    def gate(self, severity: torch.Tensor | None = None) -> torch.Tensor:
        """The gate, optionally FiLM-modulated by a scalar severity in [0, 1].

        Returns `gate_shape` when unconditioned, `(B, *gate_shape[-1:])` when
        conditioned. FiLM acts on the channel axis only and broadcasts across
        groups: a 24x1024 per-layer FiLM would be 49k outputs for no obvious
        gain.
        """
        logit = self.gate_logit
        if self.film is not None and severity is not None:
            scale, shift = self.film(severity.reshape(-1, 1)).chunk(2, dim=-1)
            logit = logit * (1 + _expand(scale, logit.unsqueeze(0))) + _expand(
                shift, logit.unsqueeze(0)
            )
        return torch.sigmoid(logit)

    def draw_noise(self, batch: int, device=None, dtype=None, generator=None) -> torch.Tensor | None:
        if not self.stochastic:
            return None
        return torch.randn(
            batch, self.noise_dim, device=device, dtype=dtype, generator=generator
        )

    def forward(
        self,
        f: torch.Tensor,
        z: torch.Tensor | None = None,
        severity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z=None` means a deterministic pass -- no noise is added at all.

        Explicit rather than auto-drawn: `identity_loss` and every test want the
        deterministic branch, and implicit sampling would make them flaky.
        """
        if z is not None and not self.stochastic:
            raise ValueError("adapter was built with noise_dim=0 but was given z")
        g = self.gate(severity)
        if severity is not None:
            g = _expand(g, f)
        for i in range(len(self.fc1)):
            h = self.fc1[i](self.norms[i](f))
            if z is not None:
                h = h + _expand(self.noise[i](z), h)
            f = f + g * self.fc2[i](self.drop(self.act(h)))
        return f

    def sample(
        self,
        f: torch.Tensor,
        k: int,
        severity: torch.Tensor | None = None,
        generator=None,
    ) -> torch.Tensor:
        """k posterior draws, stacked on a new leading axis -> (k, B, *shape).

        A deterministic adapter returns k identical copies, so callers need no
        branch; `AdaptedDetector` still forces k=1 there to avoid the waste.
        """
        outs = []
        for _ in range(k):
            z = self.draw_noise(f.shape[0], device=f.device, dtype=f.dtype, generator=generator)
            outs.append(self(f, z=z, severity=severity))
        return torch.stack(outs)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, gate_shape={self.gate_shape}, "
            f"noise_dim={self.noise_dim}, film={self.film is not None}"
        )
