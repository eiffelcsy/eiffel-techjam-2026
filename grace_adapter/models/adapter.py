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

`gate_shape` is chosen by `grace_adapter.models.factory.build_adapter`, so nothing else
in the project branches on layout. The `(L, D)` gate is also the interpretability
output: mean it over `D` and you have "how much correction does each encoder
block need", per degradation.

Four requirements, in priority order:

1. **Identity at initialization**, exactly. The last projection of every block is
   zero-initialized, so the adapter returns its input bit-for-bit whatever the
   gate or the severity conditioning happen to be. Without this a
   clean-AUC change is unattributable: did the adapter fail, or merely perturb?
2. **Gated residual.** The adapter proposes a correction; a learned gate decides
   how much to apply. Ungated, it over-corrects clean inputs -- the failure mode
   that trades away the number people actually quote.
3. **Layout-agnostic**, as above.

One optional input, a no-op when absent and identity-preserving at init:

* **`severity` -- FiLM on the gate logit.** One mildly JPEG-ed image and one
  six-operation wreck need different correction magnitudes. Zero-initialized, so
  conditioning starts as a no-op. This scalar is the thin conditioning that
  `grace_adapter.models.prompts` (FUTURE) replaces with a soft mixture over learnable
  degradation prompts; keeping it behind one call site makes that a drop-in.

Log `gate().mean()` every step. It should climb off its init (0.018 at the
default `GATE_INIT`) and plateau around 0.1-0.5. Saturating at 1.0 is
over-correction; sitting at init means the alignment term never moved the gate
at all.
"""

import torch
import torch.nn as nn

GATE_INIT = -4.0
"""sigmoid(-4) ~= 0.018. Belt to the zero-init's braces: the adapter starts as a
near-no-op even if a future change breaks the exact-identity guarantee.

Nothing pins it at exactly -4: the two constraints are "small" (so a broken
identity guarantee is still a near-no-op) and "not zero" (so the gate, and in the
ladder everything behind `tap_gate`, has gradient from step 0). Anywhere in
roughly -3 to -5 satisfies both, and -4 is the round pick inside that band. It is
the DEFAULT, not the only value: `AdapterConfig.gate_init` sweeps it, which is
what `configs/train/dinov3_sweep_gate_-3.yaml` exists to do."""


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
    severity_film : enable FiLM conditioning of the gate on a scalar severity
    gate_init   : logit the gate starts at. `GATE_INIT` unless a run is
                  deliberately sweeping it -- see the constant's note.
    """

    def __init__(
        self,
        dim: int,
        gate_shape: tuple[int, ...] | None = None,
        bottleneck: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.0,
        severity_film: bool = False,
        gate_init: float = GATE_INIT,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        self.dim = dim
        self.gate_shape = (dim,) if gate_shape is None else tuple(gate_shape)

        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_blocks))
        self.fc1 = nn.ModuleList(nn.Linear(dim, bottleneck) for _ in range(n_blocks))
        self.fc2 = nn.ModuleList(nn.Linear(bottleneck, dim) for _ in range(n_blocks))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Zero-init the *last* projection of each block: the correction is
        # identically zero at t=0 regardless of gate or severity, so
        # `test_identity_at_init` passes exactly rather than approximately.
        for layer in self.fc2:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.gate_init = float(gate_init)
        self.gate_logit = nn.Parameter(torch.full(self.gate_shape, self.gate_init))

        self.film = nn.Linear(1, 2 * dim) if severity_film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    @property
    def reads_taps(self) -> bool:
        """False here, True for `grace_adapter.models.ladder.LadderAdapter`.

        Every call site passes `taps=` unconditionally and this decides whether
        they are used or refused. The alternative -- an `isinstance` check in
        the training loop, the losses and the adapted detector -- would put the
        ladder's existence in four places instead of one.
        """
        return False

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

    def _side_input(self, taps: torch.Tensor | None) -> torch.Tensor | None:
        """Summarize the taps ONCE per forward, or None for no ladder.

        Hoisted out of the block loop because the summary -- a LayerNorm and a
        projection per tap -- does not depend on `i`, and recomputing it inside
        an `n_blocks` loop would cost `n_blocks` times the tap arithmetic to
        produce the same tensor.
        """
        return None

    def _side(self, i: int, side: torch.Tensor) -> torch.Tensor:
        """Block `i`'s read of the tap summary. Unreachable unless
        `_side_input` was overridden to return something."""
        raise NotImplementedError(f"{type(self).__name__} has no side pathway")

    def forward(
        self,
        f: torch.Tensor,
        severity: torch.Tensor | None = None,
        taps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`taps` is refused rather than ignored: a config that renders a tap
        cache and then trains a plain adapter would otherwise burn the render
        silently and report an honest-looking number for the wrong model.
        """
        if taps is not None and not self.reads_taps:
            raise ValueError(
                f"{type(self).__name__} has no ladder but was given taps of shape "
                f"{tuple(taps.shape)}. Build a LadderAdapter (set `adapter.taps` in "
                f"the train config) or stop passing them."
            )
        g = self.gate(severity)
        if severity is not None:
            g = _expand(g, f)
        side = self._side_input(taps)
        for i in range(len(self.fc1)):
            h = self.fc1[i](self.norms[i](f))
            if side is not None:
                h = h + _expand(self._side(i, side), h)
            f = f + g * self.fc2[i](self.drop(self.act(h)))
        return f

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, gate_shape={self.gate_shape}, "
            f"gate_init={self.gate_init:g}, film={self.film is not None}"
        )
