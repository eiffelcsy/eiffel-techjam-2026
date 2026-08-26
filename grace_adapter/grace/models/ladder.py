"""FUTURE -- ladder / multi-seam adapter. BLUEPRINT ONLY, nothing implemented.

The trunk is forwarded once anyway, so every intermediate activation is already
computed and thrown away. Hook 4-6 of them, run a tiny projection on each, and
fuse the results into the correction proposed at the seam -- side-tuning /
Ladder Side-Tuning, applied to feature restoration rather than to task transfer.

Why it should work here specifically: degradation does not corrupt the trunk
uniformly. JPEG destroys the high-frequency traces that early blocks encode;
blur and resize change what late blocks see semantically. A correction computed
only from the final features has to infer, from the output, which stage the
damage entered at. A ladder observes it directly.

The RINE per-layer gate (`grace.models.factory.gate_shape_for`) already
anticipates this for one detector whose trunk happens to expose every block.
Generalizing it turns "which CLIP blocks need correction under blur vs.
compression" from a RINE footnote into a result that holds across the zoo.

Prerequisites, in order:

1. `SplitDetector.taps()` returns the tap names, and `trunk` grows a way to
   return them. The hook exists today and returns `()` for every split.
2. Taps must be cached like features are, or the ladder forfeits the
   no-trunk-in-the-loop property that makes training minutes rather than hours.
   `CacheSpec.taps` already carries the field, so taps become *additional views*
   in an existing cache directory rather than a format change -- this was the
   one forward-compatibility decision worth making before any bytes were
   written.
3. Storage. k taps multiply the degraded cache by roughly k. For RINE this is
   the binding constraint, not compute; sub-sample taps or reduce `n_epochs`.

Open question the experiments should answer before this is built: does the
per-layer gate profile from the `layers` split already vary by transform? If it
does not, the premise is weak and the ladder is unlikely to repay its storage.
"""

import torch.nn as nn

from grace.splits.base import FeatureSpec, SplitDetector


class LadderAdapter(nn.Module):
    """FUTURE. Correction at the seam, informed by intermediate taps.

    Sketch of the intended shape:

        corr = base_block(f_deg)
        for name, tap in taps.items():
            corr = corr + gate[name] * proj[name](pool(tap))
        return f_deg + g * corr

    Each `proj` is a LayerNorm + Linear from the tap's width to `dim`; each tap
    gets its own gate, zero-initialized so the ladder starts as the plain
    adapter and the identity guarantee is unaffected.
    """

    def __init__(self, spec: FeatureSpec, taps: tuple[str, ...], **cfg):
        raise NotImplementedError("FUTURE -- see module docstring")


def build_ladder(split: SplitDetector, spec: FeatureSpec, cfg) -> LadderAdapter:
    """FUTURE."""
    raise NotImplementedError("FUTURE -- see module docstring")
