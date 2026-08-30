"""GRACE -- Gated Residual Adapter for Clean-feature Estimation.

A tiny adapter, trained without labels, that maps the frozen trunk's features
of a *degraded* image back onto its features of the *clean* image. The detector
never moves; the adapter is spliced between its trunk and its head.

    logit = head(adapter(trunk(x_degraded)))     ~= head(trunk(x_clean))

Only the branch's own model code lives here now -- the architecture
(`models/`) and the detector wrapper that splices it into the eval harness
(`detectors/`). Everything else (the trunk/head seam, the training loop, the
feature cache, the harness itself) lives in `eval/`, `train/`,
`preprocessing/`, and `load_data/` at the repo root, which this package is a
layer on top of. Nothing here changes a detector, and the adapted detector
re-enters the harness as an ordinary `FrozenDetector`, so every Day-1 number
is recomputed by the same code path.

See README.md for the blueprint.
"""

__version__ = "0.1.0"
