"""Which degradation each image gets in each epoch -- as a pure function.

This module is what makes the degraded-feature cache possible, so it is worth
being explicit about the argument.

The harness already draws degradations deterministically. `Condition.__call__`
keys every draw on `stable_seed(index, level, replicate, seed)` -- a blake2b
hash, never a global RNG counter -- so a recipe is a *pure function* of the image
identity and the condition, identical across runs, machines and worker
processes.

Training needs a different corruption of the same image every epoch. The harness
already has a field for exactly that: `replicate`, whose entire purpose is "an
independent re-draw over the same images". So:

    epoch == replicate

and nothing new has to be invented. Epoch 7's degradation of image 412 is
computable now, on any machine, without having run epochs 0-6 -- which is
precisely the condition for rendering every epoch offline, ahead of time.

Two draws per (image, epoch), in this order:

    1. the composition level  L ~ level_weights, keyed on (index, epoch, seed)
    2. the recipe at that level -- the harness's own draw, keyed on
       (index, L, epoch, seed), unchanged

Level 0 (clean) stays in the mix at ~15%: on those steps the alignment target
equals the input and the correct behaviour is to do nothing. That implicit
identity constraint is the only thing anchoring the adapter to a no-op on clean
inputs, which is why the level-0 share is not dropped to zero.

Requires the one-line generalisation of `Condition.sample_recipe`'s guard from
`if self.level < 2` to `if not self.grid` -- see README section 11. Without it,
training levels 0 and 1 silently produce no degradation at all.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np

from pipeline.degrade.conditions import LEVELS, Condition, Recipe
from pipeline.degrade.ops import TRANSFORMS
from pipeline.utils.seeding import stable_seed

DEFAULT_LEVEL_WEIGHTS = {0: 0.15, 1: 0.35, 2: 0.30, 3: 0.20}
"""Note the 0-indexing: the harness's L0 is clean, L1 single, L2 pair, L3 multi.
The day-2 plan wrote these 1-indexed; they are the same four numbers."""

VAL_EPOCH_OFFSET = 10_000
"""Validation epochs are numbered from here, so their `replicate` values can
never collide with a training epoch's. Held-out degradations are then disjoint
draws from the same distribution -- the only defensible way to ask whether the
adapter learned the corruption family or just these E samples of it."""

MAX_STEPS = max(spec["n_transforms"][1] for spec in LEVELS.values())
"""Deepest composition the grid can produce (5, at L3). The severity target's
depth term is normalised by this rather than by a hardcoded constant."""


@dataclass(frozen=True)
class EpochSchedule:
    """(image index, epoch) -> the degradation to apply. Deterministic, stateless.

    Held by the cache writer (to render epochs), the cache reader (to verify the
    cache matches the schedule it was asked for) and the live-mode dataset (to
    degrade on the fly). One definition, three consumers.
    """

    grid: dict[str, tuple]
    level_weights: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_LEVEL_WEIGHTS)
    )
    seed: int = 0

    def __post_init__(self):
        unknown = set(self.grid) - set(TRANSFORMS)
        if unknown:
            raise ValueError(f"grid names unregistered transforms: {sorted(unknown)}")
        bad = set(self.level_weights) - set(LEVELS)
        if bad:
            raise ValueError(f"level_weights has levels outside {sorted(LEVELS)}: {sorted(bad)}")
        total = sum(self.level_weights.values())
        if total <= 0:
            raise ValueError("level_weights must sum to something positive")
        levels = tuple(sorted(self.level_weights))
        object.__setattr__(self, "_levels", np.array(levels))
        object.__setattr__(
            self,
            "_probs",
            np.array([self.level_weights[l] / total for l in levels], dtype=float),
        )
        object.__setattr__(
            self, "_flat", tuple((n, tuple(p)) for n, p in sorted(self.grid.items()))
        )

    def level_for(self, index: int, epoch: int) -> int:
        """Weighted draw over `level_weights`, keyed on (index, epoch, seed)."""
        rng = np.random.default_rng(stable_seed(index, epoch, self.seed, "level"))
        return int(rng.choice(self._levels, p=self._probs))

    def condition_for(self, index: int, epoch: int) -> Condition:
        """The harness Condition for this cell, with `replicate=epoch`.

        `grid` is always populated, even at L0/L1: that is what makes the level a
        *distribution over recipes* rather than the fixed OFAT grid point the
        evaluation sweep uses.
        """
        level = self.level_for(index, epoch)
        return Condition(
            id=f"train/L{level}",
            level=level,
            replicate=epoch,
            grid=self._flat,
            seed=self.seed,
        )

    def recipe_for(self, index: int, epoch: int) -> Recipe:
        return self.condition_for(index, epoch).sample_recipe(index)

    def apply(self, img, index: int, epoch: int):
        """(degraded image, Recipe). The only place degradation is applied."""
        return self.condition_for(index, epoch)(img, index)

    def severity_for(self, index: int, epoch: int) -> float:
        """Corruption severity in [0, 1], exact and free.

        Grids are ordered mild -> severe (`pipeline.degrade.ops`), so a step's
        severity is its parameter's normalised rank within its own grid. Combined
        with composition depth, weighted equally:

            0.5·(n_steps / MAX_STEPS)  +  0.5·mean(rank)

        Single-valued grids (the photometric transforms and `center_crop`) get
        0.5 rather than 0: there is no milder or harsher setting to rank against,
        and scoring them 0 would say a brightness lift is as gentle as no
        degradation at all.
        """
        return self.severity_of(self.recipe_for(index, epoch))

    def severity_of(self, recipe: Recipe) -> float:
        if not recipe.steps:
            return 0.0
        ranks = []
        for step in recipe.steps:
            values = self.grid[step.transform]
            if len(values) == 1:
                ranks.append(0.5)
            else:
                ranks.append(values.index(step.param) / (len(values) - 1))
        depth = len(recipe.steps) / MAX_STEPS
        return float(np.clip(0.5 * depth + 0.5 * float(np.mean(ranks)), 0.0, 1.0))

    def fingerprint(self) -> str:
        """Hash of grid + weights + seed.

        Stored in the cache spec and asserted on load. Change a parameter value
        in configs/degradations.yaml and every cached degraded feature is
        silently wrong; this is the tripwire.
        """
        payload = repr((self._flat, sorted(self.level_weights.items()), self.seed))
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def val_epochs(n: int) -> range:
    """Epoch ids reserved for held-out degradations. See VAL_EPOCH_OFFSET."""
    return range(VAL_EPOCH_OFFSET, VAL_EPOCH_OFFSET + n)
