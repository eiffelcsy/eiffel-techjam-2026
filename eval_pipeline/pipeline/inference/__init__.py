"""Standalone inference: a directory of images -> P(generated) per image."""

from pipeline.inference.predict import predict_dir, write_predictions

__all__ = ["predict_dir", "write_predictions"]
