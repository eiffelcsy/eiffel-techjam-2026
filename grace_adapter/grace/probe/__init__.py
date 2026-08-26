"""Stage 0: fit the classifier the rest of the pipeline holds frozen.

GRACE never trains a detector -- that is the claim. But the PoC detector
(`pipeline.detectors.dinov3.DINOv3MLPDetector`) is a frozen DINOv3 trunk plus a
head that has to come from somewhere, and "somewhere" is here: one linear probe
on clean features, run once, before anything else in this package executes.

It lives in `grace/` rather than in the evaluation harness because the harness
loads and scores detectors and never trains them, and because this is
scaffolding for the PoC rather than a description of a published model. The
harness's view of the result is a finished checkpoint at a path in a config.
"""

from grace.probe.train import extract_features, train_probe

__all__ = ["extract_features", "train_probe"]
