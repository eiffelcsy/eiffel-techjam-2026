"""Evaluation conditions, organised into four composition levels.

    L0  clean          the reference. Sets the threshold and the retention
                       denominator.
    L1  single         one transform at one parameter, the full deterministic
                       grid (19 conditions). One-factor-at-a-time: every
                       result is attributable to exactly one cause.
    L2  pair           two distinct transforms, sampled per image from the
                       eleven and their discrete parameter values, applied in
                       random order.
    L3  multi          three to five distinct transforms, sampled the same way.

L1 is a controlled sweep; L2/L3 are Monte-Carlo samples of the composition
space (there are far too many combinations to enumerate). The pair with L1 is
the point: L1 says *why* a detector fails, L2/L3 say *how much it will fail in
deployment*, and the gap between the two is the interaction effect -- see
`pipeline.eval.metrics.interaction_gap`.

Sampling rules for L2/L3:
  - transforms are drawn without replacement, so no image gets JPEG twice
  - each drawn transform gets a parameter drawn uniformly from its own grid
  - application order is shuffled: these transforms do not commute
    (blur-then-JPEG and JPEG-then-blur leave different traces)
  - the draw is keyed on (image index, level, replicate), so the degraded eval
    set is identical across runs and across detectors
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from pipeline.degrade.ops import TRANSFORMS
from pipeline.utils.seeding import stable_seed

LEVELS = {
    0: dict(name="clean",  n_transforms=(0, 0)),
    1: dict(name="single", n_transforms=(1, 1)),
    2: dict(name="pair",   n_transforms=(2, 2)),
    3: dict(name="multi",  n_transforms=(3, 5)),
}


@dataclass(frozen=True)
class Step:
    """One transform at one of its discrete parameter values."""

    transform: str      # "jpeg"
    param: Any          # 30

    def label(self) -> str:
        """"jpeg/quality=30" -- the unit of the per-transform breakdown."""
        return f"{self.transform}/{TRANSFORMS[self.transform].param_name}={self.param}"


@dataclass(frozen=True)
class Recipe:
    """The ordered list of steps actually applied to one image.

    Logged per image so the composed levels can be analysed after the fact:
    which pairs hurt most, whether order mattered, whether a single step
    dominates the drop.
    """

    steps: tuple[Step, ...]

    def label(self) -> str:
        """"gaussian_blur/sigma=1.0 + jpeg/quality=50", in application order."""
        return " + ".join(s.label() for s in self.steps) or "clean"

    def transforms(self) -> tuple[str, ...]:
        return tuple(s.transform for s in self.steps)


@dataclass(frozen=True)
class Condition:
    """One column of the results table.

    L0/L1 conditions are fixed: every image gets the same recipe, carried in
    `steps`. L2/L3 conditions are distributions: `steps` is empty and each
    image draws its own recipe from `grid`, so one condition covers many
    combinations across the eval set.
    """

    id: str             # "clean" | "jpeg/quality=30" | "L2" | "L3"
    level: int
    replicate: int = 0  # >0 only for composed levels, see n_replicates
    steps: tuple[Step, ...] = ()
    grid: tuple[tuple[str, tuple], ...] = ()
    seed: int = 0

    def sample_recipe(self, index: int) -> Recipe:
        """Deterministic draw keyed on (index, level, replicate).

        A condition carrying no `grid` is a *fixed* recipe and returns it: that
        is every L0 and L1 condition the evaluation sweep builds, where L1 is the
        19-point one-factor-at-a-time grid and each point is its own condition.

        A condition carrying a grid is a *distribution* and draws from it, at any
        level -- `LEVELS[level]["n_transforms"]` already says how many transforms
        each level composes, including (0, 0) for clean and (1, 1) for single. So
        `Condition(level=1, grid=...)` means "one transform drawn at random",
        which is what a training schedule wants and what the eval sweep, having
        no grid, never asks for. See grace_adapter/grace/cache/schedule.py.
        """
        if not self.grid:
            return Recipe(self.steps)

        lo, hi = LEVELS[self.level]["n_transforms"]
        rng = np.random.default_rng(
            stable_seed(index, self.level, self.replicate, self.seed)
        )
        n = min(int(rng.integers(lo, hi + 1)), len(self.grid))
        # permutation, not choice: draws without replacement *and* fixes the
        # application order in one step, since these transforms do not commute.
        picked = rng.permutation(len(self.grid))[:n]
        steps = tuple(
            Step(self.grid[j][0], self.grid[j][1][int(rng.integers(len(self.grid[j][1])))])
            for j in picked
        )
        return Recipe(steps)

    def __call__(self, img: Image.Image, index: int) -> tuple[Image.Image, Recipe]:
        """Apply the recipe; return the image and what was done to it."""
        recipe = self.sample_recipe(index)
        rng = np.random.default_rng(
            stable_seed(index, self.level, self.replicate, self.seed, "apply")
        )
        for step in recipe.steps:
            img = TRANSFORMS[step.transform].fn(img, step.param, rng)
        return img, recipe


def build_conditions(
    grid: dict[str, tuple],
    levels: list[int],
    n_replicates: int = 1,
    seed: int = 0,
) -> list[Condition]:
    """Expand the transform grid into the ordered list of conditions.

    `n_replicates` re-draws L2/L3 over the same images with a different seed.
    Each replicate is an independent sample of the composition space, so it
    both tightens the confidence interval on the composed-level AUC and widens
    coverage of the combination space -- cheaper than growing the eval set,
    and it keeps the clean scores (and therefore the pairing) fixed.
    """
    flat = tuple((name, tuple(params)) for name, params in grid.items())
    conditions: list[Condition] = []

    if 0 in levels:
        conditions.append(Condition(id="clean", level=0, seed=seed))
    if 1 in levels:
        for name, params in flat:
            for p in params:
                step = Step(name, p)
                conditions.append(
                    Condition(id=step.label(), level=1, steps=(step,), seed=seed)
                )
    for level in (2, 3):
        if level in levels:
            conditions.extend(
                Condition(id=f"L{level}", level=level, replicate=r, grid=flat, seed=seed)
                for r in range(n_replicates)
            )
    return conditions


def load_grid(path: str | Path, transforms: list[str] | None = None) -> dict[str, tuple]:
    """Read the transform -> parameter grid out of configs/degradations.yaml."""
    with Path(path).open() as f:
        spec = yaml.safe_load(f)["grid"]
    unknown = set(spec) - set(TRANSFORMS)
    if unknown:
        raise ValueError(f"grid names unregistered transforms: {sorted(unknown)}")
    return {
        name: tuple(entry["values"])
        for name, entry in spec.items()
        if transforms is None or name in transforms
    }
