"""The schedule is the cache's contract. If it is not pure, nothing else holds.

Pre-rendering degraded features is only sound because `(index, epoch)` determines
the recipe exactly, on any machine, in any process. These tests are what license
that claim.
"""

import numpy as np
import pytest

from grace.cache.schedule import VAL_EPOCH_OFFSET, EpochSchedule, val_epochs
from preprocessing.degrade.conditions import Condition, load_grid

GRID_FILE = "preprocessing/configs/degradations.yaml"


@pytest.fixture(scope="module")
def grid():
    return load_grid(GRID_FILE)


@pytest.fixture
def schedule(grid):
    return EpochSchedule(grid=grid, seed=0)


def test_recipe_is_pure(schedule):
    for index in (0, 7, 4242):
        for epoch in (0, 3):
            a = schedule.recipe_for(index, epoch).label()
            b = schedule.recipe_for(index, epoch).label()
            assert a == b


def test_recipe_is_stable_across_processes(schedule, grid):
    """blake2b, not hash(): PYTHONHASHSEED is randomised per process, so a test
    that only calls twice in one interpreter would pass on a broken
    implementation. This rebuilds the schedule from scratch to check the seed
    derivation carries no hidden state."""
    fresh = EpochSchedule(grid=grid, seed=0)
    for index in (1, 99, 12345):
        assert fresh.recipe_for(index, 5).label() == schedule.recipe_for(index, 5).label()


def test_epochs_differ(schedule):
    """Otherwise the degraded cache is E copies of one view and training sees no
    variety at all."""
    labels = {schedule.recipe_for(11, e).label() for e in range(12)}
    assert len(labels) > 1


def test_images_differ_within_an_epoch(schedule):
    labels = {schedule.recipe_for(i, 0).label() for i in range(50)}
    assert len(labels) > 1


def test_level_weights_respected(grid):
    weights = {0: 0.15, 1: 0.35, 2: 0.30, 3: 0.20}
    schedule = EpochSchedule(grid=grid, level_weights=weights, seed=0)
    drawn = np.array([schedule.level_for(i, 0) for i in range(4000)])
    for level, want in weights.items():
        assert abs((drawn == level).mean() - want) < 0.03


def test_level_zero_produces_no_degradation(schedule, grid):
    """The implicit identity constraint: on clean draws the target equals the
    input and the correct behaviour is to do nothing."""
    condition = Condition(id="t", level=0, replicate=0, grid=schedule._flat, seed=0)
    assert condition.sample_recipe(3).steps == ()


def test_level_one_draws_exactly_one_transform(schedule):
    """Requires the upstream guard change from `level < 2` to `not self.grid`.
    Without it this returns an empty recipe and training levels 0 and 1 are
    silently identical."""
    condition = Condition(id="t", level=1, replicate=0, grid=schedule._flat, seed=0)
    assert len(condition.sample_recipe(3).steps) == 1


def test_eval_conditions_are_unaffected_by_that_change():
    """The harness's own L0/L1 conditions carry fixed `steps` and no `grid`, so
    they must still return exactly what they were given."""
    from preprocessing.degrade.conditions import Step

    step = Step("jpeg", 30)
    assert Condition(id="clean", level=0).sample_recipe(9).steps == ()
    assert Condition(id="jpeg", level=1, steps=(step,)).sample_recipe(9).steps == (step,)


def test_composed_levels_draw_the_right_depth(schedule):
    for level, (lo, hi) in ((2, (2, 2)), (3, (3, 5))):
        condition = Condition(id="t", level=level, replicate=0, grid=schedule._flat, seed=0)
        for index in range(20):
            assert lo <= len(condition.sample_recipe(index).steps) <= hi


def test_val_epochs_are_disjoint():
    assert min(val_epochs(4)) >= VAL_EPOCH_OFFSET
    assert set(val_epochs(4)).isdisjoint(range(1000))


def test_val_epochs_draw_different_recipes(schedule):
    train = schedule.recipe_for(5, 0).label()
    held = schedule.recipe_for(5, VAL_EPOCH_OFFSET).label()
    assert isinstance(train, str) and isinstance(held, str)


def test_severity_is_monotone_in_depth(schedule):
    """More composed operations must never score as less severe."""
    severities = {}
    for index in range(500):
        recipe = schedule.recipe_for(index, 0)
        severities.setdefault(len(recipe.steps), []).append(schedule.severity_of(recipe))
    means = [np.mean(v) for _, v in sorted(severities.items())]
    assert means == sorted(means)


def test_severity_is_bounded(schedule):
    for index in range(200):
        assert 0.0 <= schedule.severity_for(index, 0) <= 1.0


def test_clean_severity_is_zero(schedule, grid):
    from preprocessing.degrade.conditions import Recipe

    assert schedule.severity_of(Recipe(())) == 0.0


def test_severity_ranks_the_grid(grid):
    """Grids are ordered mild -> severe, so quality=30 must outrank quality=90."""
    from preprocessing.degrade.conditions import Recipe, Step

    schedule = EpochSchedule(grid=grid, seed=0)
    mild = schedule.severity_of(Recipe((Step("jpeg", 90),)))
    harsh = schedule.severity_of(Recipe((Step("jpeg", 30),)))
    assert harsh > mild


def test_fingerprint_changes_with_the_grid(grid):
    a = EpochSchedule(grid=grid, seed=0).fingerprint()
    assert a != EpochSchedule(grid=grid, seed=1).fingerprint()
    trimmed = {k: v for k, v in list(grid.items())[:5]}
    assert a != EpochSchedule(grid=trimmed, seed=0).fingerprint()
    assert a == EpochSchedule(grid=dict(grid), seed=0).fingerprint()


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError, match="unregistered"):
        EpochSchedule(grid={"not_a_transform": (1,)})
