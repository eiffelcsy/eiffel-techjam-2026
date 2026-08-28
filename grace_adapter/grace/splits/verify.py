"""The split contract, checked at construction rather than trusted.

    head(trunk(x)) == detector(x)

Every zoo split composes modules that live in a vendored repo under
`third_party/`, cloned by hand and not present in this tree. A split whose head
composition is subtly wrong does not crash -- it produces plausible logits from a
model that was never benchmarked, and every retention number computed from it is
a comparison against nothing.

So each split runs this on one random batch in `__init__`. The cost is one
forward pass; the alternative is finding out on day 5.
"""

import torch

TOL = 1e-4


def verify_split(split, batch: int = 2, tol: float = TOL) -> None:
    """Raise unless `head(trunk(x))` reproduces `detector(x)` on random input.

    The probe is random noise rather than a real image on purpose: it needs no
    dataset, and a composition error shows up on any input.
    """
    detector = split.detector
    first = next(detector.parameters(), None)
    device = first.device if first is not None else torch.device("cpu")
    x = torch.randn(batch, *_probe_shape(split), device=device)

    with torch.no_grad():
        expected = detector(x)
        try:
            actual = split.head(split.trunk(x))
        except Exception as e:                       # noqa: BLE001 - re-raised with context
            raise RuntimeError(_message(split, f"head(trunk(x)) raised {e!r}")) from e

    if actual is None:
        raise RuntimeError(_message(split, "head(trunk(x)) returned None"))
    if actual.shape != expected.shape:
        raise RuntimeError(
            _message(split, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        )
    gap = (actual - expected).abs().max().item()
    if gap > tol:
        raise RuntimeError(_message(split, f"max |difference| = {gap:.3g} > {tol:g}"))


def verify_taps(split, batch: int = 2, tol: float = TOL) -> None:
    """Raise unless the split's three tap declarations agree with each other.

    Checked, in order, because each failure is silent in a different way:

      * `taps()` and `tap_spec()` disagree on count -- the cache would be
        rendered at one width and the names in `spec.json` would describe
        another, so a later "which block was tap 3?" is unanswerable.
      * emitted taps do not match `tap_spec()` -- `ShardWriter` catches this on
        the first batch, hours after `--dry-run` printed a size computed from
        the spec.
      * **`trunk_with_taps(x)[0] != trunk(x)`** -- the one that corrupts
        results rather than crashing. The ladder arm would be trained and scored
        on a seam that no baseline was ever measured at, and every retention
        comparison against the plain adapter would be off by whatever the two
        paths happen to differ by.
    """
    names, spec = split.taps(), split.tap_spec()
    if not names:
        return
    if spec is None:
        raise RuntimeError(
            f"{type(split).__name__}.taps() names {names} but tap_spec() is None. "
            f"A split that exposes taps must declare their shape."
        )
    if spec.shape[0] != len(names):
        raise RuntimeError(
            f"{type(split).__name__} declares {len(names)} tap name(s) {names} but "
            f"tap_spec() has {spec.shape[0]} group(s) {spec.shape}. The names index "
            f"the group axis; they cannot disagree."
        )

    first = next(split.detector.parameters(), None)
    device = first.device if first is not None else torch.device("cpu")
    x = torch.randn(batch, *_probe_shape(split), device=device)
    with torch.no_grad():
        plain = split.trunk(x)
        f, taps = split.trunk_with_taps(x)

    if taps is None:
        raise RuntimeError(
            f"{type(split).__name__}.trunk_with_taps() returned no taps although "
            f"taps() names {names}."
        )
    if tuple(taps.shape[1:]) != spec.shape:
        raise RuntimeError(
            f"{type(split).__name__} emitted taps of shape {tuple(taps.shape[1:])} "
            f"per image but tap_spec() declares {spec.shape}."
        )
    gap = (f - plain).abs().max().item()
    if gap > tol:
        raise RuntimeError(
            f"{type(split).__name__}.trunk_with_taps() does not reproduce trunk(): "
            f"max |difference| = {gap:.3g} > {tol:g}. The tapped path must return "
            f"the SAME seam features -- the cache, the frozen head and every "
            f"baseline number were built against those. Usually this means the "
            f"tapped path skipped a final norm or pooled differently."
        )


def _probe_shape(split) -> tuple[int, ...]:
    """A plausible input shape for this detector, from its own preprocessing."""
    size = getattr(split.detector, "input_size", 224)
    return (3, size, size)


def _message(split, problem: str) -> str:
    modules = getattr(split, "head_modules", lambda: {})()
    listing = "\n".join(f"    {n}: {type(m).__name__}" for n, m in modules.items())
    return (
        f"{type(split).__name__} does not reproduce {split.name}: {problem}.\n"
        f"`_head_forward` must compose the trained modules exactly as the "
        f"vendored repo's own forward does. Trainable modules found on the "
        f"model:\n{listing or '    (none found)'}\n"
        f"Fix `_head_forward` against the clone, then re-run. Do NOT pass "
        f"verify=False to get past this: the split would score a model that was "
        f"never benchmarked."
    )
