"""Exact search: the answer key for small instances (SPEC 10.2).

Uniform-cost search (Dijkstra) over `PrepState.canonical_key()`, where an edge costs that step's
exact scalar cost and the goal is meeting all demand. Optionally A* with an admissible
heuristic. Returns `optimal` only when the search finished within its budget.

Why this is possible at all: `rules.apply` is a pure function (no env to copy) and canonical
keys drop droplet uids, so states that differ only by relabelling collapse into one. Without
that the search would never terminate.

**`include_step`** (default true) puts the step count in the key. It has to, in general: with it
excluded, a state settled cheaply at step 10 would block a pricier arrival at step 3 that still
had budget left to finish, so the search could return a suboptimal cost or wrongly report
infeasible. Excluding it is faster and is safe only when the step limit cannot bind.

**Branch and bound.** The search starts from an incumbent plan (the naive baseline by default)
and prunes any node whose `g + h` already reaches the incumbent's cost. If nothing cheaper is
found, the incumbent is proved optimal. This is what makes the search finish: the reachable
space is finite but vast, since cheap buffer dilutions can wander far into fine compositions
that no target needs.

Costs are exact `Rational`s throughout, so two plans are compared without any rounding.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from meda.core.actions import Action
from meda.core.composition import ZERO, Rational
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG
from meda.core.problem import Instance
from meda.core.rules import apply, legal_actions, replay
from meda.core.state import CanonicalKey, PrepState
from meda.solvers.base import SolveBudget, SolveResult
from meda.solvers.baselines.naive import naive_plan

Heuristic = Callable[[PrepState, Instance], Rational]
HeuristicName = Literal["zero", "fluid_bound"]


def zero_heuristic(state: PrepState, instance: Instance) -> Rational:
    """Plain Dijkstra."""
    return ZERO


def fluid_bound_heuristic(weights: CostWeights) -> Heuristic:
    """Admissible lower bound from conservation of each fluid.

    Every droplet still to be delivered for target t holds exactly
    `droplet_volume * composition[f]` units of fluid f, and that fluid can only come from what is
    already on chip or from new reservoir draws. So future draws of fluid f are at least
    `needed[f] - on_chip[f]`, and the remaining cost is at least the weighted sum of those.
    Mix and waste costs are ignored, which only makes the bound smaller (so it stays admissible).
    """

    def heuristic(state: PrepState, instance: Instance) -> Rational:
        num_fluids = instance.num_fluids
        needed = [ZERO] * num_fluids
        for target, remaining in zip(instance.targets, state.remaining_demand, strict=True):
            if not remaining:
                continue
            units = remaining * target.droplet_volume
            for f in range(num_fluids):
                if target.composition.num[f]:
                    needed[f] = needed[f] + Rational.of(
                        units * target.composition.num[f], target.composition.den
                    )
        for droplet in state.droplets:
            for f in range(num_fluids):
                if droplet.composition.num[f]:
                    needed[f] = needed[f] - Rational.of(
                        droplet.volume * droplet.composition.num[f], droplet.composition.den
                    )
        total = ZERO
        for fluid, short in zip(instance.fluids, needed, strict=True):
            if short > 0:
                total = total + weights.fluid_weight(fluid.kind) * short
        return total

    return heuristic


@dataclass(frozen=True, slots=True)
class SearchStats:
    nodes_expanded: int
    nodes_generated: int
    peak_frontier: int
    exhausted: bool  # the reachable space was searched out (no budget stop)


@dataclass(slots=True)
class _Node:
    """The settled representative of a canonical key, and how we got here."""

    state: PrepState
    parent: CanonicalKey | None
    action: Action | None


class ExactSearch:
    """`Solver` that proves the optimal plan cost for an instance.

    `heuristic`: "fluid_bound" (A*, admissible) or "zero" (plain Dijkstra). Both must return the
    same cost; a test checks that on every instance of `tiny_exact_v1`.
    """

    def __init__(
        self,
        weights: CostWeights,
        heuristic: HeuristicName = "fluid_bound",
        include_step: bool = True,
        use_incumbent: bool = True,
    ) -> None:
        self.weights = weights
        self.heuristic_name = heuristic
        self.include_step = include_step
        self.use_incumbent = use_incumbent
        self.name = "exact_search"
        self._heuristic: Heuristic = (
            fluid_bound_heuristic(weights) if heuristic == "fluid_bound" else zero_heuristic
        )
        self.last_stats: SearchStats | None = None

    def solve(self, instance: Instance, budget: SolveBudget | None = None) -> SolveResult:
        budget = budget or SolveBudget()
        start = time.perf_counter()
        plan, cost, stats = self.search(instance, budget)
        wall = time.perf_counter() - start
        self.last_stats = stats

        if plan is None:
            status: Literal["infeasible", "timeout"] = (
                "infeasible" if stats.exhausted else "timeout"
            )
            return SolveResult(
                instance.instance_id,
                self.name,
                status,
                None,
                None,
                None,
                wall,
                stats.nodes_expanded,
            )
        _, totals, deltas = replay(instance, plan)
        assert cost is not None
        return SolveResult(
            instance_id=instance.instance_id,
            producer=self.name,
            status="optimal",
            dag=MixingDAG.from_deltas(instance, deltas, totals, self.name),
            totals=totals,
            cost=cost,
            wall_time=wall,
            nodes_expanded=stats.nodes_expanded,
        )

    def incumbent(self, instance: Instance) -> tuple[tuple[Action, ...], Rational] | None:
        """A feasible plan to bound the search with (the naive baseline), if it has one."""
        if not self.use_incumbent:
            return None
        plan = naive_plan(instance)
        if plan is None:
            return None
        _, totals, _ = replay(instance, plan)
        return plan, self.weights.scalar(totals, instance.fluids)

    def search(
        self, instance: Instance, budget: SolveBudget | None = None
    ) -> tuple[tuple[Action, ...] | None, Rational | None, SearchStats]:
        """The optimal plan and its exact cost, or None with the reason in the stats."""
        budget = budget or SolveBudget()
        deadline = None if budget.max_seconds is None else time.perf_counter() + budget.max_seconds
        step_in_key = self.include_step
        heuristic = self._heuristic

        best = self.incumbent(instance)
        best_plan: tuple[Action, ...] | None = best[0] if best else None
        bound: Rational | None = best[1] if best else None

        root = PrepState.initial(instance)
        root_key = root.canonical_key(step_in_key)
        dist: dict[CanonicalKey, Rational] = {root_key: ZERO}
        nodes: dict[CanonicalKey, _Node] = {root_key: _Node(root, None, None)}
        counter = 0
        frontier: list[tuple[Rational, int, CanonicalKey]] = [
            (heuristic(root, instance), counter, root_key)
        ]
        expanded = generated = 0
        peak = 1

        while frontier:
            if budget.max_nodes is not None and expanded >= budget.max_nodes:
                return None, None, SearchStats(expanded, generated, peak, exhausted=False)
            if deadline is not None and time.perf_counter() > deadline:
                return None, None, SearchStats(expanded, generated, peak, exhausted=False)

            priority, _, key = heapq.heappop(frontier)
            node = nodes[key]
            g = dist[key]
            # A stale heap entry: a cheaper route to this key was found after it was pushed.
            # Re-opening like this keeps A* optimal even if the heuristic is not consistent.
            if priority > g + heuristic(node.state, instance):
                continue
            if node.state.is_success:
                return (
                    self._reconstruct(nodes, key),
                    g,
                    SearchStats(expanded, generated, peak, exhausted=False),
                )
            expanded += 1

            for action in legal_actions(node.state, instance):
                nxt, step_cost, _ = apply(node.state, action, instance)
                generated += 1
                cost = g + self.weights.scalar(step_cost, instance.fluids)
                nkey = nxt.canonical_key(step_in_key)
                if nkey in dist and not cost < dist[nkey]:
                    continue
                dist[nkey] = cost
                nodes[nkey] = _Node(nxt, key, action)
                counter += 1
                heapq.heappush(frontier, (cost + heuristic(nxt, instance), counter, nkey))
                peak = max(peak, len(frontier))

        # The space was searched out with nothing cheaper: the incumbent is optimal.
        stats = SearchStats(expanded, generated, peak, exhausted=True)
        if best_plan is not None:
            return best_plan, bound, stats
        return None, None, stats

    @staticmethod
    def _reconstruct(nodes: dict[CanonicalKey, _Node], goal: CanonicalKey) -> tuple[Action, ...]:
        """Walk the parent chain back to the root.

        The stored action for a key was taken from its parent's settled state, and `apply` is
        deterministic, so replaying the list from the initial state reproduces exactly these
        states (the caller replays it to build the DAG, which validates the result).
        """
        actions: list[Action] = []
        key: CanonicalKey | None = goal
        while key is not None:
            node = nodes[key]
            if node.action is not None:
                actions.append(node.action)
            key = node.parent
        actions.reverse()
        return tuple(actions)
