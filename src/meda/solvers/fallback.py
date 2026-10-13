"""Cost to finish: the truncation penalty (SPEC 10.3).

An **upper** estimate of the cost to finish from a state, and the cost of a plan that really
exists: waste everything on chip, then build each remaining droplet from the reservoirs with the
naive construction. Because it is the cost of a feasible completion, it can never be less than
the optimal remaining cost, which is what the truncation penalty needs — otherwise an unfinished
plan could look cheaper than a finished one (the AlphaTensor step-limit idea).

The penalty is hypothetical: it answers "what would finishing cost with a known method", so it
is computed on a relaxed instance and does not have to fit inside the step limit that was just
exceeded.

Scope: the naive construction (10.4) must be able to build the remaining targets. That covers
every dilution stage. A mixture whose denominator exceeds the mix limit has no naive plan, and
this raises an error naming the instance rather than inventing a number — rule 9. Mixture
stages need a general baseline first (reserved for M9).
"""

from __future__ import annotations

from dataclasses import replace

from meda.core.composition import ZERO, Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.core.state import PrepState
from meda.solvers.baselines.naive import naive_plan

RELAXED_STEP_LIMIT = 100_000


class UnsupportedFallbackError(ValueError):
    """The naive construction cannot build the remaining targets of this instance."""


class NaiveCostToFinish:
    """`cost_to_finish(state, instance)` for `SamplePrepEnv`.

    The naive part depends only on the remaining demand, so it is cached per
    (instance, remaining demand); the waste part is added per state. Picklable, so it works with
    `AsyncVectorEnv`.
    """

    def __init__(self, weights: CostWeights) -> None:
        self.weights = weights
        self._cache: dict[tuple[str, tuple[int, ...]], Rational] = {}

    def __call__(self, state: PrepState, instance: Instance) -> Rational:
        on_chip = sum(d.volume for d in state.droplets)
        waste = self.weights.w_waste * on_chip
        return waste + self.remaining_cost(state.remaining_demand, instance)

    def remaining_cost(self, remaining: tuple[int, ...], instance: Instance) -> Rational:
        """Naive cost of building `remaining` droplets per target from the reservoirs."""
        if not any(remaining):
            return ZERO
        key = (instance.instance_id, remaining)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        targets = tuple(
            replace(target, id=i, demand=demand)
            for i, (target, demand) in enumerate(
                (t, d) for t, d in zip(instance.targets, remaining, strict=True) if d > 0
            )
        )
        relaxed = Instance(
            f"{instance.instance_id}#finish",
            instance.fluids,
            targets,
            instance.chip,
            RELAXED_STEP_LIMIT,
        )
        plan = naive_plan(relaxed)
        if plan is None:
            raise UnsupportedFallbackError(
                f"instance '{instance.instance_id}': no cost-to-finish estimate. The naive "
                f"construction cannot build the remaining targets on chip '{instance.chip.name}' "
                f"(a mixture needs denominator <= {instance.chip.max_mix_volume}; every "
                f"denominator needs factors within [{instance.chip.min_mix_volume}, "
                f"{instance.chip.max_mix_volume}]). Mixture stages need a general baseline "
                f"(reserved for M9); until then use a different cost_to_finish."
            )
        _, totals, _ = replay(relaxed, plan)
        cost = self.weights.scalar(totals, relaxed.fluids)
        self._cache[key] = cost
        return cost

    def supports(self, instance: Instance) -> bool:
        """Whether this instance can be penalised at all (checked once, before training)."""
        try:
            self.remaining_cost(tuple(t.demand for t in instance.targets), instance)
        except UnsupportedFallbackError:
            return False
        return True
