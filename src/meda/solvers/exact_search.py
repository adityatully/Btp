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

**Branch and bound.** The search starts from an incumbent plan (the cheaper of naive and split tree)
and prunes any node whose `g + h` already reaches the incumbent's cost. If nothing cheaper is
found, the incumbent is proved optimal. This is what makes the search finish: the reachable
space is finite but vast, since cheap buffer dilutions can wander far into fine compositions
that no target needs.

**Lower bound on a budget stop.** The search then returns status `timeout` with
`lower_bound = min(f_min, incumbent)`, where `f_min` is the smallest `g + h` on the frontier.
With an admissible `h`, some state of an optimal plan always waits on the frontier with
`f <= C*`, so `f_min <= C*`. Stale heap entries only carry a larger `f`, and states pruned by the
incumbent have `f >= incumbent`, which is why the incumbent caps the bound. The result also
carries the best plan known at that point: a goal generated more cheaply than the incumbent
replaces it (and tightens the pruning), so the upper end improves too.

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
from meda.solvers.fallback import cheapest_construction

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
    stopped: bool = False  # a budget stopped the search; the plan returned is only the best known
    lower_bound: Rational | None = None  # set when `stopped`: a proved bound on the optimum


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

        status: Literal["optimal", "infeasible", "timeout"] = (
            "timeout" if stats.stopped else "infeasible" if plan is None else "optimal"
        )
        if plan is None:
            return SolveResult(
                instance.instance_id,
                self.name,
                status,
                None,
                None,
                None,
                wall,
                stats.nodes_expanded,
                stats.lower_bound,
            )
        _, totals, deltas = replay(instance, plan)
        assert cost is not None
        return SolveResult(
            instance_id=instance.instance_id,
            producer=self.name,
            status=status,
            dag=MixingDAG.from_deltas(instance, deltas, totals, self.name),
            totals=totals,
            cost=cost,
            wall_time=wall,
            nodes_expanded=stats.nodes_expanded,
            lower_bound=stats.lower_bound,
        )

    def incumbent(self, instance: Instance) -> tuple[tuple[Action, ...], Rational] | None:
        """A feasible plan to bound the search with: the cheaper of the naive and split-tree
        constructions (10.3), if either builds one within the step limit."""
        if not self.use_incumbent:
            return None
        return cheapest_construction(instance, self.weights)

    def search(
        self, instance: Instance, budget: SolveBudget | None = None
    ) -> tuple[tuple[Action, ...] | None, Rational | None, SearchStats]:
        """The optimal plan and its exact cost, or None with the reason in the stats.

        If a budget stops the search (`stats.stopped`), the plan and cost returned are the best
        known (None if there is none) and `stats.lower_bound` bounds the optimum from below."""
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
            if (budget.max_nodes is not None and expanded >= budget.max_nodes) or (
                deadline is not None and time.perf_counter() > deadline
            ):
                f_min = frontier[0][0]
                lower = f_min if bound is None or f_min < bound else bound
                stats = SearchStats(expanded, generated, peak, False, True, lower)
                return best_plan, bound, stats

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
                f = cost + heuristic(nxt, instance)
                # Branch and bound: reaching the incumbent's cost cannot improve on it.
                if bound is not None and not f < bound:
                    continue
                dist[nkey] = cost
                nodes[nkey] = _Node(nxt, key, action)
                counter += 1
                heapq.heappush(frontier, (f, counter, nkey))
                peak = max(peak, len(frontier))
                if nxt.is_success:  # a cheaper plan than the incumbent: it becomes the incumbent
                    best_plan, bound = self._checked(instance, nodes, nkey, cost), cost

        # The space was searched out with nothing cheaper: the incumbent is optimal.
        stats = SearchStats(expanded, generated, peak, exhausted=True)
        if best_plan is not None:
            return best_plan, bound, stats
        return None, None, stats

    def _checked(
        self,
        instance: Instance,
        nodes: dict[CanonicalKey, _Node],
        goal: CanonicalKey,
        cost: Rational,
    ) -> tuple[Action, ...]:
        """The plan to `goal`, replayed to confirm it succeeds at exactly `cost`."""
        plan = self._reconstruct(nodes, goal)
        final, totals, _ = replay(instance, plan)
        if not final.is_success or self.weights.scalar(totals, instance.fluids) != cost:
            raise RuntimeError(
                f"instance '{instance.instance_id}': reconstructed plan does not replay to "
                f"success at cost {cost}"
            )
        return plan

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
