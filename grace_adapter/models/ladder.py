"""The ladder -- correction at the seam, informed by intermediate taps.

    corr_i = fc2_i( act( fc1_i(LN(f)) + rung_i(summary(taps)) ) )
    y      = f + g * sum_i corr_i

The trunk is forwarded once anyway, so every intermediate activation is already
computed and thrown away. Keep a few, summarize them, and let that summary
modulate the correction the adapter proposes at the seam -- side-tuning applied
to feature restoration rather than to task transfer.

Why it works here, in one number
--------------------------------
Degradation does not corrupt the trunk uniformly. JPEG destroys the
high-frequency traces early blocks encode; blur and resize change what late
blocks see semantically. A correction computed only from the final features has
to infer, from the output, which stage the damage entered at. A ladder observes
it directly.

That was the premise, and it is now measured rather than argued. Asked to
identify which of nine L1 transforms hit an image (chance 0.111), the seam alone
scores **0.376** and five taps score **0.896** -- see `DEFAULT_TAP_BLOCKS` in
`eval.splits.dinov3` for the full table and the tap selection it drove. The
plain adapter is very nearly blind to the thing it is trying to undo.

Where the taps enter, and why not where the blueprint said
----------------------------------------------------------
The original sketch added a projected tap directly to the correction:

    corr = base(f) + gate_k * proj_k(tap_k)

This adds it to the **bottleneck** instead, which is the same place `z` enters.
Two reasons. A tap term added to `corr` can only *translate* the correction -- it
is a bias, constant in `f` -- whereas the ladder's job is to say what *kind* of
correction this damage calls for, which means modulating a function of `f`, not
offsetting it. Side signals perturb the proposed correction; they do not bypass
it.

Identity at initialization is unaffected, and not by luck: `fc2` is zero-init in
`GatedResidualAdapter`, so the correction is identically zero however the
bottleneck is perturbed. The ladder cannot break the guarantee that makes a
clean-AUC change attributable. `tap_gate` still starts at the seam gate's own
init (`GATE_INIT`, ~0.018, or whatever `adapter.gate_init` sets) rather than at
zero -- belt to the same braces, and a hard zero there would put a
second multiplicative zero in front of `tap_proj` and leave it gradient-starved
for the first stretch of training.

The parameter budget
--------------------
GRACE's claim rests on the adapter being small: "if a large adapter turns out to
be needed, the claim is false and the result is far less interesting". So the
ladder does not get a projection per tap per block. It gets

    tap_norms   one LayerNorm per tap          K * 2 * tap_in
    tap_proj    ONE projection, shared         tap_in * tap_dim
    tap_gate    one gate vector per tap        K * tap_dim
    rung        one read per adapter block     n_blocks * K*tap_dim * bottleneck

which at the PoC's K=5, tap_in=768, tap_dim=64, bottleneck=256, n_blocks=2 is
~0.22M against the base adapter's ~0.4M. Sharing `tap_proj` across taps is the
load-bearing choice: the ladder learns what damage *looks like* once, and
`tap_gate` plus the tap's position in the concatenated summary carry *where* it
happened.

`tap_gate()` is the interpretability output, and it is the figure the `layers`
gate promised for RINE without needing a `layers` detector: mean it over
`tap_dim` and you have "how much does the correction lean on block k", per
degradation. Log it.
"""

import torch
import torch.nn as nn

from grace_adapter.models.adapter import GatedResidualAdapter
from eval.splits.base import FeatureSpec, SplitDetector


class LadderAdapter(GatedResidualAdapter):
    """`GatedResidualAdapter` plus a read of the trunk's intermediate taps.

    Everything the plain adapter does, it still does: same gate, same residual,
    same severity path, same exact identity at init. `taps=None`
    makes it behave as the plain adapter, which is what `AdaptedDetector` relies
    on when a ladder checkpoint is scored on a split that emits no taps.

    Parameters
    ----------
    tap_spec  : `(K, tap_in)`, from `SplitDetector.tap_spec()`
    tap_dim   : width each tap is projected to before the rungs read it
    tap_names : for `extra_repr` and for reading `tap_gate()` back as a figure
    **cfg     : passed to `GatedResidualAdapter`
    """

    def __init__(
        self,
        dim: int,
        tap_spec: FeatureSpec,
        tap_dim: int = 64,
        tap_names: tuple[str, ...] = (),
        **cfg,
    ):
        super().__init__(dim=dim, **cfg)
        n_taps, tap_in = tap_spec.shape
        if tap_names and len(tap_names) != n_taps:
            raise ValueError(
                f"tap_spec declares {n_taps} tap(s) but tap_names has "
                f"{len(tap_names)}: {tap_names}"
            )
        self.n_taps = n_taps
        self.tap_in = tap_in
        self.tap_dim = tap_dim
        self.tap_names = tuple(tap_names) or tuple(f"tap{k}" for k in range(n_taps))

        # Per-tap norm, shared projection. The norm is per-tap because the taps
        # arrive already normalized by the trunk's own final norm but at
        # genuinely different scales (~9-25 max-abs across depth on DINOv3
        # ViT-S/16); the projection is shared because damage is damage.
        self.tap_norms = nn.ModuleList(nn.LayerNorm(tap_in) for _ in range(n_taps))
        self.tap_proj = nn.Linear(tap_in, tap_dim)
        self.tap_gate_logit = nn.Parameter(
            torch.full((n_taps, tap_dim), self.gate_init)
        )

        bottleneck = self.fc1[0].out_features
        self.rung = nn.ModuleList(
            nn.Linear(n_taps * tap_dim, bottleneck, bias=False)
            for _ in range(len(self.fc1))
        )

    @property
    def reads_taps(self) -> bool:
        return True

    def tap_gate(self) -> torch.Tensor:
        """`(K, tap_dim)` in [0, 1]. Mean over the last axis for the figure:
        "how much does the correction lean on tap k". Unconditioned -- the
        severity FiLM acts on the seam gate, which is where a scalar belongs."""
        return torch.sigmoid(self.tap_gate_logit)

    def tap_weights(self) -> dict[str, float]:
        """`tap_gate()` as a named scalar per tap, for `summary.json` and W&B."""
        with torch.no_grad():
            g = self.tap_gate().mean(dim=-1)
        return {name: float(g[k]) for k, name in enumerate(self.tap_names)}

    def tap_drift(self, taps: torch.Tensor) -> torch.Tensor:
        """`(B, K, tap_in)` -> `(B, K, tap_dim)`: the gated per-tap read.

        The ladder's own description of "what damage is this, at block k",
        computed from the degraded image alone. `_side_input` flattens this into
        the bottleneck; it is exposed separately because the discrepancy head
        wants the same tensor UNflattened, so it can take one norm per tap and
        recover a per-block damage profile.

        That profile is what the `layers` layout gives GRACE-D for free and a
        `vector` seam like DINOv3's does not: Δ is one vector however deep the
        damage entered, so without this the head sees a single drift norm and
        cannot say WHERE the image was hit. Free here -- the ladder computes this
        tensor for its own forward pass either way.
        """
        if taps.ndim != 3 or tuple(taps.shape[1:]) != (self.n_taps, self.tap_in):
            raise ValueError(
                f"expected taps of shape (B, {self.n_taps}, {self.tap_in}), got "
                f"{tuple(taps.shape)}. The adapter and the cache disagree about "
                f"the tap set -- check `adapter.taps` against the cache's spec.json."
            )
        normed = torch.stack(
            [self.tap_norms[k](taps[:, k]) for k in range(self.n_taps)], dim=1
        )
        return self.tap_proj(normed) * self.tap_gate()

    def _side_input(self, taps: torch.Tensor | None) -> torch.Tensor | None:
        """`(B, K, tap_in)` -> `(B, K * tap_dim)`, once per forward.

        `taps=None` is allowed and means "no ladder this pass", so a ladder
        checkpoint still runs -- as the plain adapter -- wherever taps are not
        available. That is a real configuration, not a fallback: it is the
        ablation that isolates what the ladder itself contributed.
        """
        if taps is None:
            return None
        return self.tap_drift(taps).flatten(1)

    def _side(self, i: int, side: torch.Tensor) -> torch.Tensor:
        return self.rung[i](side)

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, taps={self.n_taps}x{self.tap_in}"
            f"->{self.tap_dim} {self.tap_names}"
        )


def build_ladder(spec: FeatureSpec, tap_spec: FeatureSpec, cfg, tap_names=()) -> LadderAdapter:
    """Built from the same `AdapterConfig` the plain adapter is, so a ladder arm
    and its control differ by the `taps` key alone."""
    from grace_adapter.models.factory import gate_shape_for

    return LadderAdapter(
        dim=spec.dim,
        tap_spec=tap_spec,
        tap_dim=cfg.tap_dim,
        tap_names=tuple(tap_names),
        gate_shape=gate_shape_for(spec, cfg.per_channel_gate),
        bottleneck=cfg.bottleneck,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        severity_film=cfg.severity_film,
        gate_init=cfg.gate_init,
    )


def tap_spec_for(split: SplitDetector, cfg) -> FeatureSpec | None:
    """The tap spec a run should build against, or None for a plain adapter.

    The split is the authority, not the config: `cfg.taps` says *whether* to
    build a ladder, the split says what shape the taps actually are. A config
    asking for taps against a split that emits none is an error here rather than
    a shape mismatch several thousand steps later.
    """
    if not getattr(cfg, "taps", False):
        return None
    tap_spec = split.tap_spec()
    if tap_spec is None:
        raise ValueError(
            f"adapter.taps is set but {type(split).__name__} emits no taps. Pass "
            f"`tap_blocks` to the split (configs name it under `split_args`), or "
            f"unset `adapter.taps`."
        )
    return tap_spec
