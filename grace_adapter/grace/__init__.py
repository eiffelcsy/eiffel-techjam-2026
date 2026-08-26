"""GRACE -- Gated Residual Adapter for Clean-feature Estimation.

A tiny adapter, trained without labels, that maps the frozen trunk's features
of a *degraded* image back onto its features of the *clean* image. The detector
never moves; the adapter is spliced between its trunk and its head.

    logit = head(adapter(trunk(x_degraded)))     ~= head(trunk(x_clean))

Layered on top of `robust-aigc-eval` (the sibling `eval_pipeline/`), which owns
the manifest, the degradation grid, and the measurement. Nothing here changes a
detector, and the adapted detector re-enters that harness as an ordinary
`FrozenDetector`, so every Day-1 number is recomputed by the same code path.

See README.md for the blueprint.
"""

__version__ = "0.1.0"
