"""Cost to finish: the truncation penalty (SPEC 10.3).

An **upper** estimate of the cost to finish from a state, and the cost of a plan that really
exists: waste everything on chip, then build each remaining target's droplets from the
reservoirs, per target with the cheaper of two constructions (`baselines.naive` and
`baselines.tree`). Because it is the cost of a feasible completion, it can never be less than
the optimal remaining cost, which is what the truncation penalty needs — otherwise an unfinished
plan could look cheaper than a finished one (the AlphaTensor step-limit idea). Taking the
cheaper of the two keeps it as close to the true cost of finishing as these constructions allow.

The penalty is hypothetical: it answers "what would finishing cost with a known method", so it
is computed on a relaxed instance and does not have to fit inside the step limit that was just
exceeded. The chip (including its storage capacity) is the instance's own.

Scope: every target whose denominator has no prime factor above M (every reachable target),
as long as the chip's storage holds the tree's depth. Outside that, this raises an error naming
the instance rather than inventing a number (rule 9).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from meda.core.actions import Action
from meda.core.composition import ZERO, Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.core.state import PrepState
from meda.solvers.baselines.naive import naive_plan
from meda.solvers.baselines.tree import tree_plan

RELAXED_STEP_LIMIT = 100_000
CACHE_LIMIT = 200_000  # entries; the cache is cleared when full (values never depend on it)
CONSTRUCTIONS: tuple[Callable[[Instance], tuple[Action, ...] | None], ...] = (
    naive_plan,
    tree_plan,
)


class UnsupportedFallbackError(ValueError):
    """No construction can build the remaining targets of this instance."""


def cheapest_construction(
    instance: Instance, weights: CostWeights
) -> tuple[tuple[Action, ...], Rational] | None:
    """The cheaper plan of naive and tree for the whole instance, with its cost, or None."""
    best: tuple[tuple[Action, ...], Rational] | None = None
    for construct in CONSTRUCTIONS:
        plan = construct(instance)
        if plan is None:
            continue
        _, totals, _ = replay(instance, plan)
        cost = weights.scalar(totals, instance.fluids)
        if best is None or cost < best[1]:
            best = (plan, cost)
    return best


class ConstructiveCostToFinish:
    """`cost_to_finish(state, instance)` for `SamplePrepEnv`.

    Both constructions build each target on its own from an empty chip, so the cost of the
    remaining demand is a sum over targets, each the cheaper construction for that target.
    It is cached per (target, remaining demand, fluid kinds, chip limits), so the same target
    on another instance reuses it and a generator that draws a fresh problem every episode
    does not grow the cache per instance; the waste part is added per state. Picklable, so it
    works with `AsyncVectorEnv`.
    """

    def __init__(self, weights: CostWeights) -> None:
        self.weights = weights
        self._cache: dict[tuple[object, ...], Rational] = {}

    def __call__(self, state: PrepState, instance: Instance) -> Rational:
        on_chip = sum(d.volume for d in state.droplets)
        waste = self.weights.w_waste * on_chip
        return waste + self.remaining_cost(state.remaining_demand, instance)

    def remaining_cost(self, remaining: tuple[int, ...], instance: Instance) -> Rational:
        """Cost of building `remaining` droplets per target from the reservoirs."""
        total = ZERO
        for target, demand in zip(instance.targets, remaining, strict=True):
            if demand:
                total = total + self._target_cost(instance, target.id, demand)
        return total

    def _target_cost(self, instance: Instance, target_id: int, demand: int) -> Rational:
        target = replace(instance.targets[target_id], id=0, experiment_id=0, demand=demand)
        chip = instance.chip
        key = (
            target,
            tuple(f.kind for f in instance.fluids),
            chip.max_mix_volume,
            chip.min_mix_volume,
            chip.min_droplet_volume,
            chip.allow_partial_take,
            chip.storage_capacity,
            chip.max_denominator,
            chip.allow_same_composition_merge,
        )
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        relaxed = Instance(
            f"{instance.instance_id}#finish-t{target_id}",
            instance.fluids,
            (target,),
            instance.chip,
            RELAXED_STEP_LIMIT,
        )
        best = cheapest_construction(relaxed, self.weights)
        if best is None:
            raise UnsupportedFallbackError(
                f"instance '{instance.instance_id}': no cost-to-finish estimate for target "
                f"{target_id} ({target.composition}) on chip '{instance.chip.name}'. Neither "
                f"construction can build it: its denominator needs every prime factor <= "
                f"max_mix_volume {instance.chip.max_mix_volume}, and the split tree needs "
                f"storage for its depth (storage_capacity {instance.chip.storage_capacity})."
            )
        if len(self._cache) >= CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = best[1]
        return best[1]

    def supports(self, instance: Instance) -> bool:
        """Whether this instance can be penalised at all (checked once, before training)."""
        try:
            self.remaining_cost(tuple(t.demand for t in instance.targets), instance)
        except UnsupportedFallbackError:
            return False
        return True
