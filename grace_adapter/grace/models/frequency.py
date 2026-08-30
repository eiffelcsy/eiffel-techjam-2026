"""The frequency enricher: read the image again, where the trunk cannot.

    f_corrected = adapter(f_degraded)                      stage 1, label-free
    fused       = f_corrected + g_hf . out_hf + g_lf . out_lf

`out_b` is a multi-head cross-attention: the corrected spatial feature is the
query, and the keys and values are 196 patch-DCT cells of the *same window* the
trunk saw, each masked to one band expert's slice of the coefficient axis.

WHY A SECOND READ OF THE IMAGE AT ALL. The adapter can only rearrange what the
trunk kept. A ViT at 224 has already resampled the window, and generation traces
are a local high-frequency phenomenon at native pixel scale -- so whatever the
resize destroyed is not in `f_degraded` for the adapter to restore, at any
capacity. The DCT branch reads the pixels the trunk threw away. That is also why
this can exceed retention 1.0 in principle and the adapter cannot: a restorer is
bounded by the clean-image score, and this is not restoring.

WHY TWO BAND EXPERTS. Blur and downscale DESTROY high-frequency energy; gaussian
noise ADDS it; JPEG does both, moving energy onto block-aligned coefficients. One
expert reading the whole spectrum has to express all of that with one set of
weights and one gate, and the gate is a scalar per channel -- it cannot open for
one kind of damage and shut for another. Two experts with their own masks, their
own K/V projections, their own output projections and their own gates can, and
E11 is the arm that says whether the split earns its parameters.

IDENTITY AT INITIALIZATION, EXACTLY. Every expert's output projection is
zero-initialized -- weight and bias -- so `fused == f_corrected` bit for bit at
step 0, for ANY frequency tokens and ANY severity. This is the same mechanism the
adapter uses (`adapter.py`'s zero-init `fc2`), and the gates at `sigmoid(-4)` are
belt to its braces rather than the mechanism itself. It is what makes E10 a
tautology instead of a nail-biter, and what makes stage 2's step-0 validation AUC
provably equal to the `+grace` arm's: any change in a reported number is
attributable to what the enricher learned, not to having wired it in.

NO LAYERNORM ON THE OUTPUT. The residual is added raw. A LayerNorm on the sum
would rescale `f_corrected` itself, which breaks the identity and, worse, feeds
the frozen head a feature space it was never fit on -- the head is the one part
of this pipeline that has to keep its provenance.
"""

import math

import torch
import torch.nn as nn

from grace.models.adapter import GATE_INIT, _expand
from pipeline.freq.dct import band_masks

MASK_INIT_LOGIT = 4.0
"""Logit magnitude the band masks start at. `sigmoid(+-4) = 0.982 / 0.018`.

Near-hard rather than hard, on purpose. `band_masks` returns exact 0/1 rectangles
and the honest parametrization of "a mask in [0, 1]" is a sigmoid, but a sigmoid
saturated at exactly 0 or 1 has no gradient -- a learnable mask pinned there could
never move, which would make `learn_masks: true` a lie. At +-4 the mask is 98%
hard and every coefficient still passes gradient to its own band boundary.

The complementarity survives it for the two-band default: `sigmoid(4) +
sigmoid(-4) == 1` exactly, so at initialization the HF and LF experts still see a
decomposition of the spectrum rather than a selection from it. That identity is
particular to two bands and is checked in `tests/test_enricher_identity.py`.
"""


def _mask_logits(patch: int, n_bands: int, channels: int) -> torch.Tensor:
    """`(n_bands, channels * patch**2)` logits whose sigmoid is `band_masks`.

    Linear in the mask value so a soft (raised-cosine) initialization keeps its
    shape rather than being squashed through an inverse sigmoid, which would
    send the 0 and 1 ends to infinity.
    """
    masks = torch.from_numpy(band_masks(patch, n_bands, channels)).float()
    return MASK_INIT_LOGIT * (2.0 * masks - 1.0)


class BandExpert(nn.Module):
    """One band's cross-attention read of the DCT cells.

    Owns everything: its mask over the coefficient axis, its K/V projections
    (which see the masked tokens, so the mask acts before any mixing), its
    query projection down from the seam, its attention, and its zero-initialized
    output projection back up to the seam width.

    The mask is applied to the TOKENS, not to the attention logits. Masking the
    logits would decide which *cells* to look at; masking the tokens decides
    which *frequencies* are visible in every cell, which is the actual split --
    a blur destroys high frequencies everywhere in the frame, not in some cells.
    """

    def __init__(
        self,
        dim: int,
        n_coeffs: int,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.0,
        mask_init: torch.Tensor | None = None,
        learn_masks: bool = True,
    ):
        super().__init__()
        if mask_init is None:
            mask_init = torch.full((n_coeffs,), MASK_INIT_LOGIT)
        if mask_init.shape != (n_coeffs,):
            raise ValueError(
                f"mask_init must be ({n_coeffs},), got {tuple(mask_init.shape)}"
            )
        self.mask_logit = nn.Parameter(mask_init.clone(), requires_grad=learn_masks)

        # LayerNorm over the coefficient axis, after masking: the DC coefficient
        # is O(1) and the top radial band is O(1e-3) even after log1p, and an
        # unnormalized projection would let DC dominate every key it produces.
        self.norm = nn.LayerNorm(n_coeffs)
        self.k_proj = nn.Linear(n_coeffs, d_model)
        self.v_proj = nn.Linear(n_coeffs, d_model)
        self.q_proj = nn.Linear(dim, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.out = nn.Linear(d_model, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def mask(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logit)

    def forward(
        self, tokens: torch.Tensor, query: torch.Tensor, pos: torch.Tensor | None
    ) -> torch.Tensor:
        """`(B, cells, coeffs)`, `(B, dim)` -> `(B, dim)`, zero at init."""
        h = self.norm(tokens * self.mask())
        k = self.k_proj(h)
        if pos is not None:
            # Position is added to the KEYS only. The query is a pooled spatial
            # feature with no cell of its own, and the values carry content that
            # a position offset would corrupt -- the attention needs to know
            # WHERE a cell is to select it, not to report a coordinate back.
            k = k + pos
        attended, _ = self.attn(
            self.q_proj(query).unsqueeze(1), k, self.v_proj(h), need_weights=False
        )
        return self.out(attended.squeeze(1))


class FrequencyEnricher(nn.Module):
    """`(f_corrected, freq_tokens, severity) -> fused`, identity at init.

    Parameters
    ----------
    dim         : seam width, `FeatureSpec.dim` (768 on the PoC detector)
    n_cells     : cells per image, `grid**2`
    n_coeffs    : coefficients per cell, `channels * patch**2`
    patch, channels : the DCT geometry behind `n_coeffs`. Needed for the band
                  masks and for `top_k`, both of which index the coefficient
                  axis by (channel, radial position) rather than flat.
    """

    def __init__(
        self,
        dim: int,
        n_cells: int,
        n_coeffs: int,
        patch: int = 8,
        channels: int = 3,
        d_model: int = 256,
        n_heads: int = 4,
        n_bands: int = 2,
        dropout: float = 0.0,
        gate_init: float = GATE_INIT,
        severity_film: bool = True,
        pos_emb: bool = True,
        top_k: int | None = None,
        learn_masks: bool = True,
    ):
        super().__init__()
        if channels * patch * patch != n_coeffs:
            raise ValueError(
                f"channels*patch^2 = {channels * patch * patch} but n_coeffs is "
                f"{n_coeffs}. The coefficient axis is channel-major over "
                f"radially-ordered blocks; the two must describe one layout."
            )
        grid = math.isqrt(n_cells)
        if grid * grid != n_cells:
            raise ValueError(f"n_cells must be a square grid, got {n_cells}")

        self.dim, self.n_cells, self.n_coeffs = int(dim), int(n_cells), int(n_coeffs)
        self.patch, self.channels, self.grid = int(patch), int(channels), grid
        self.n_bands = int(n_bands)
        self.top_k = top_k

        # Selection happens at READ time over a fully rendered view, so E13's
        # top-k sweep costs no re-render. That is the whole reason the cache
        # stores full coefficients: the view count is the resumable knob and the
        # coefficient set is not, so the irreversible one is taken at its safest
        # setting and the ablation is paid for in the model.
        select = _top_k_index(patch, channels, top_k)
        self.register_buffer("coeff_index", select, persistent=False)
        width = self.n_coeffs if select is None else int(select.numel())

        masks = _mask_logits(patch, n_bands, channels)
        if select is not None:
            masks = masks[:, select]
        self.experts = nn.ModuleList(
            BandExpert(
                dim, width, d_model=d_model, n_heads=n_heads, dropout=dropout,
                mask_init=masks[b], learn_masks=learn_masks,
            )
            for b in range(n_bands)
        )

        # 2D by construction: one embedding per row plus one per column, summed
        # at the cell. 2*grid*d_model parameters instead of grid^2*d_model, and
        # it gives the attention a separable coordinate rather than 196
        # unrelated vectors.
        if pos_emb:
            self.row_emb = nn.Parameter(torch.zeros(grid, d_model))
            self.col_emb = nn.Parameter(torch.zeros(grid, d_model))
            nn.init.normal_(self.row_emb, std=0.02)
            nn.init.normal_(self.col_emb, std=0.02)
        else:
            self.row_emb = self.col_emb = None

        self.gate_init = float(gate_init)
        self.gate_logit = nn.Parameter(torch.full((n_bands, dim), self.gate_init))
        # Per band and per channel, zero-initialized like the adapter's, so
        # conditioning starts as a no-op and the identity does not depend on it.
        self.film = nn.Linear(1, 2 * n_bands * dim) if severity_film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    # -- the pieces, exposed so diagnostics and tests can read them ------------

    def pos(self) -> torch.Tensor | None:
        """`(n_cells, d_model)` in row-major cell order, matching `cell_pool`'s
        `reshape(grid*grid, ...)`. None when position embeddings are off."""
        if self.row_emb is None:
            return None
        return (self.row_emb[:, None, :] + self.col_emb[None, :, :]).reshape(
            self.n_cells, -1
        )

    def gates(self, severity: torch.Tensor | None = None) -> torch.Tensor:
        """`(n_bands, dim)`, or `(B, n_bands, dim)` when FiLM-conditioned."""
        logit = self.gate_logit
        if self.film is not None and severity is not None:
            scale, shift = self.film(severity.reshape(-1, 1)).chunk(2, dim=-1)
            shape = (-1, self.n_bands, self.dim)
            logit = logit * (1 + scale.reshape(shape)) + shift.reshape(shape)
        return torch.sigmoid(logit)

    def tokens(self, freq: torch.Tensor) -> torch.Tensor:
        """Cast, shape-check and (for E13) select down the coefficient axis."""
        if freq.ndim != 3 or freq.shape[1] != self.n_cells:
            raise ValueError(
                f"expected frequency tokens (B, {self.n_cells}, {self.n_coeffs}), "
                f"got {tuple(freq.shape)}"
            )
        if freq.shape[2] != self.n_coeffs:
            raise ValueError(
                f"this enricher was built over {self.n_coeffs} coefficients but "
                f"got {freq.shape[2]}. The cache was rendered under a different "
                f"DCT protocol -- check freq_sha."
            )
        out = freq.float()
        return out if self.coeff_index is None else out[:, :, self.coeff_index]

    def forward(
        self,
        f: torch.Tensor,
        freq: torch.Tensor,
        severity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`f` is the CORRECTED feature (the adapter's output), not the raw one.

        The enricher sits after stage 1 and before the frozen head, so what it
        enriches is the best spatial estimate available. Handing it `f_degraded`
        instead would make it a second restorer trained with labels, which is a
        different method and not this one.
        """
        tokens = self.tokens(freq)
        pos = self.pos()
        gates = self.gates(severity)
        out = f
        for b, expert in enumerate(self.experts):
            # Unconditioned gates are `(dim,)` and already broadcast; only the
            # FiLM-conditioned `(B, dim)` needs singleton axes inserted, and only
            # then against a seam wider than a vector. Same rule as the adapter's.
            gate = gates[b]
            if gates.ndim == 3:
                gate = _expand(gates[:, b], f)
            out = out + gate * expert(tokens, f, pos)
        return out


def _top_k_index(patch: int, channels: int, top_k: int | None) -> torch.Tensor | None:
    """Coefficient positions of the `k` lowest radial frequencies, per channel.

    Returns None for "keep everything", which is the fast path -- no indexing at
    all rather than an identity gather.

    Per channel, not flat, because the coefficient axis is channel-major: a flat
    `[:k]` would keep the whole red channel and none of the others, which is a
    colour ablation wearing a frequency ablation's name.
    """
    per = patch * patch
    if top_k is None or top_k >= per:
        return None
    return torch.tensor(
        [c * per + j for c in range(channels) for j in range(top_k)], dtype=torch.long
    )
