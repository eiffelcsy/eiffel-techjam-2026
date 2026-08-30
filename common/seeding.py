"""Deterministic seeding.

Degradations must be reproducible: the same image index always gets the same
corruption, in every run, forever.
"""

import hashlib
import random

import numpy as np


def stable_seed(*parts: int | str) -> int:
    """Hash the parts into a stable 32-bit seed. Never a global counter.

    Uses blake2b rather than the builtin hash(): PYTHONHASHSEED is randomised
    per process, so hash() would give a different degraded set on every run and
    silently break cross-run and cross-detector comparability.
    """
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
