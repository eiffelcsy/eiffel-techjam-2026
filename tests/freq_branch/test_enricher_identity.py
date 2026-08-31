"""At initialization the enricher must be EXACTLY `f_corrected`, not approximately.

The same argument the adapter's identity test makes, one stage later and with one
more input to sweep over. A module that perturbs the corrected features at step 0
makes every later number unattributable: a change in AUC could be what the
enricher learned, or it could be the noise of having wired a cross-attention
block into the seam.

So the guarantee is swept over RANDOM FREQUENCY TOKENS and RANDOM SEVERITY, not
merely the zero case. Both are places a future change could break it quietly --
a bias on an output projection, a LayerNorm on the residual, a gate that stopped
multiplying a zero -- and the zero case would catch none of them.
"""

import pytest
import torch

from train.config import EnricherConfig, FreqConfig
from freq_branch.models.factory import build_enricher, load_enricher, save_enricher
from eval.splits.base import FeatureSpec

SPEC = FeatureSpec(layout="vector", shape=(32,))
FREQ = FreqConfig(enabled=True, patch=4, grid=3)   # (9, 48)


def _enricher(**kwargs):
    return build_enricher(
        SPEC, FREQ.feature(), EnricherConfig(**kwargs),
        patch=FREQ.patch, channels=FREQ.channels,
    )


def _inputs(batch: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(batch, SPEC.dim, generator=g)
    # log1p'd DCT magnitudes are non-negative and span orders of magnitude;
    # `abs` and a scale keep the fixture in that regime rather than testing the
    # identity against a distribution the module will never see.
    freq = torch.randn(batch, *FREQ.shape, generator=g).abs() * 3.0
    severity = torch.rand(batch, generator=g)
    return f, freq, severity


def test_identity_at_init():
    enricher = _enricher()
    f, freq, _ = _inputs()
    assert torch.equal(enricher(f, freq), f)


@pytest.mark.parametrize("seed", range(5))
def test_identity_holds_for_any_frequency_and_any_severity(seed):
    """Exact equality, not `allclose`: the output projections are zero, so the
    correction is identically zero and nothing rounds."""
    enricher = _enricher(severity_film=True, dropout=0.3)
    f, freq, severity = _inputs(seed=seed)
    assert torch.equal(enricher(f, freq, severity), f)
    assert torch.equal(enricher(f, freq, torch.zeros_like(severity)), f)
    assert torch.equal(enricher(f, freq * 1000, severity), f)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bands": 1},                       # E11
        {"n_bands": 4},
        {"pos_emb": False},                   # E13
        {"top_k": 4},                         # E13
        {"learn_masks": False},
        {"severity_film": False},
        {"gate_init": -1.0},
        {"d_model": 64, "n_heads": 8},
    ],
    ids=lambda k: ",".join(f"{a}={b}" for a, b in k.items()),
)
def test_every_ablation_keeps_the_identity(kwargs):
    """Each of these is a shipped experiment arm, and each must start from the
    same place as the reference arm -- otherwise the arms are not comparable to
    it or to each other."""
    enricher = _enricher(**kwargs)
    f, freq, severity = _inputs(seed=7)
    assert torch.equal(enricher(f, freq, severity), f)


def test_the_gates_start_shut():
    enricher = _enricher(gate_init=-4.0)
    gates = enricher.gates()
    assert gates.shape == (2, SPEC.dim)
    assert torch.allclose(gates, torch.sigmoid(torch.tensor(-4.0)), atol=1e-6)


def test_update_is_zero_at_init():
    """The UN-GATED expert sum is identically zero too -- the aux and
    orthogonality terms therefore start at zero, and every later number is a
    delta from a DCT branch that contributed nothing."""
    enricher = _enricher()
    f, freq, _ = _inputs()
    assert torch.equal(enricher.update(f, freq), torch.zeros_like(f))


def test_fused_is_f_plus_the_gated_expert_sum():
    """The read/update/forward refactor must preserve the fusion exactly:
    `update` is the pre-gate sum of the reads, and `forward` adds the GATED
    reads to f."""
    enricher = _enricher(severity_film=False)
    f, freq, _ = _inputs(seed=2)
    with torch.no_grad():
        for expert in enricher.experts:
            expert.out.weight.normal_()
            expert.out.bias.normal_()
    reads = enricher.reads(f, freq)
    assert reads.shape == (f.shape[0], enricher.n_bands, f.shape[1])
    assert torch.equal(enricher.update(f, freq), reads.sum(dim=1))
    gates = enricher.gates()
    manual = f + sum(gates[b] * reads[:, b] for b in range(enricher.n_bands))
    assert torch.allclose(enricher(f, freq), manual, atol=1e-6)


def test_aux_head_reads_the_un_gated_update():
    """The aux head's input is the DCT branch's own read, so its logit depends
    on the experts and not on the gate or the fusion -- which is what makes
    `auc_aux` a measurement of the frequency signatures alone."""
    from freq_branch.models.frequency import EnricherAuxHead

    enricher = _enricher()
    head = EnricherAuxHead(SPEC.dim, hidden=8)
    f, freq, _ = _inputs()
    logit = head(enricher.update(f, freq))
    assert logit.shape == (f.shape[0],)
    assert not torch.isnan(logit).any()


def test_fused_logit_is_identity_at_init():
    """`logit_main + beta * aux(update)` with beta=0 is exactly `logit_main` --
    the step-0 identity the fused-logit arm relies on (beta leaves zero only if
    the frequency signal has no decision value)."""
    from freq_branch.models.frequency import EnricherFusedLogit

    fused = EnricherFusedLogit(SPEC.dim, hidden=8)
    assert float(fused.beta.detach()) == 0.0
    update = torch.randn(6, SPEC.dim)
    logit_main = torch.randn(6)
    assert torch.equal(fused(logit_main, update), logit_main)


def test_the_two_bands_decompose_the_spectrum_at_init():
    """`sigmoid(+4) + sigmoid(-4) == 1` exactly, so the HF and LF masks still sum
    to one at every coefficient. The band split is therefore a decomposition of
    the spectrum rather than a selection from it: nothing the model could read is
    dropped by the split, only routed. Particular to two bands -- see
    `MASK_INIT_LOGIT`."""
    enricher = _enricher(n_bands=2)
    total = sum(expert.mask() for expert in enricher.experts)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_the_bands_are_actually_different():
    """A guard against the masks being initialized identically, which would make
    E11 (two experts vs one) a comparison of a model against itself."""
    hf, lf = (expert.mask() for expert in _enricher(n_bands=2).experts)
    assert float((hf - lf).abs().max().detach()) > 0.9


def test_top_k_keeps_the_lowest_frequencies_of_every_channel():
    """Per channel, not flat. The coefficient axis is channel-major, so a flat
    `[:k]` would keep all of the first channel and none of the others -- a colour
    ablation wearing a frequency ablation's name."""
    enricher = _enricher(top_k=3)
    per = FREQ.patch * FREQ.patch
    assert enricher.coeff_index.tolist() == [
        c * per + j for c in range(FREQ.channels) for j in range(3)
    ]


def test_top_k_beyond_the_coefficient_count_is_the_fast_path():
    """"Keep everything" must be no indexing at all, not an identity gather."""
    assert _enricher(top_k=FREQ.patch * FREQ.patch).coeff_index is None
    assert _enricher(top_k=None).coeff_index is None


def test_a_mismatched_token_width_is_refused():
    """Same cell count, different coefficient axis. The shapes are close enough
    to load and the band masks would be indexed against the wrong frequencies."""
    enricher = _enricher()
    f, _, _ = _inputs()
    with pytest.raises(ValueError, match="coefficients"):
        enricher(f, torch.rand(f.shape[0], FREQ.shape[0], FREQ.shape[1] + 8))


def test_checkpoint_round_trip_preserves_the_function(tmp_path):
    """A checkpoint must rebuild into the same module with no reference to the
    run that wrote it -- what makes `configs/detectors/*+grace-freq.yaml` able to
    name a path and nothing else."""
    enricher = _enricher(n_bands=3, top_k=5, pos_emb=True)
    # Move it off the identity first: a round trip that only ever compares two
    # identities proves nothing about the weights.
    with torch.no_grad():
        for expert in enricher.experts:
            expert.out.weight.normal_()
            expert.out.bias.normal_()
    f, freq, severity = _inputs(seed=3)
    want = enricher(f, freq, severity)

    path = tmp_path / "enricher.pt"
    save_enricher(
        path, enricher, SPEC, FREQ.feature(),
        EnricherConfig(n_bands=3, top_k=5, pos_emb=True), FREQ,
    )
    assert torch.equal(load_enricher(str(path), SPEC, FREQ.feature())(f, freq, severity), want)


def test_loading_against_a_different_dct_protocol_is_refused(tmp_path):
    """`(196, 192)` from an 8x8x3 render and `(196, 192)` from another geometry
    are the same shape and different features. Nothing downstream would notice."""
    enricher = _enricher()
    path = tmp_path / "enricher.pt"
    save_enricher(path, enricher, SPEC, FREQ.feature(), EnricherConfig(), FREQ)
    other = FreqConfig(enabled=True, patch=4, grid=4).feature()   # (16, 48)
    with pytest.raises(ValueError, match="frequency tokens"):
        load_enricher(str(path), SPEC, other)
