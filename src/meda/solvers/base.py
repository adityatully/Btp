"""`Solver` protocol, `SolveBudget`, `SolveResult` (SPEC 10.1).

Agents are evaluated through the same `SolveResult` (see `meda.agents.base.run_episode`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from meda.core.actions import Action, Discard, Mix, Source
from meda.core.composition import Rational
from meda.core.costs import StepCost
from meda.core.dag import MixingDAG
from meda.core.problem import Instance
from meda.core.rules import apply
from meda.core.state import PrepState

# optimal / feasible / timeout / infeasible: solvers. failed: an agent episode that did not meet
# all demand (that is not a proof of infeasibility).
SolveStatus = Literal["optimal", "feasible", "timeout", "infeasible", "failed"]


@dataclass(frozen=True, slots=True)
class SolveBudget:
    max_nodes: int | None = None
    max_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class SolveResult:
    """`dag`, `totals` and `cost` describe the produced plan (None if there is none).

    `cost` is the exact scalar plan cost under the weights used (no truncation penalty). On a
    `timeout`, the plan (if any) is only the best known, and `lower_bound` bounds the optimum."""

    instance_id: str
    producer: str
    status: SolveStatus
    dag: MixingDAG | None
    totals: StepCost | None
    cost: Rational | None
    wall_time: float
    nodes_expanded: int = 0
    lower_bound: Rational | None = None  # a proved bound on the optimum (exact search timeout)


class Solver(Protocol):
    name: str

    def solve(self, instance: Instance, budget: SolveBudget) -> SolveResult: ...


@dataclass(slots=True)
class PlanBuilder:
    """Emits a plan by applying actions through `core.rules.apply`.

    Every plan built this way is legal by construction: an illegal step raises
    `IllegalActionError` instead of being recorded. Shared by the solvers and by the
    generators, which use it to build each instance's feasibility witness.
    """

    instance: Instance
    state: PrepState
    actions: list[Action]

    @staticmethod
    def start(instance: Instance) -> PlanBuilder:
        return PlanBuilder(instance, PrepState.initial(instance), [])

    def do(self, action: Action) -> None:
        self.state, _, _ = apply(self.state, action, self.instance)
        self.actions.append(action)

    def mix(self, a: Source, k: int, b: Source, l: int) -> int:
        """Apply the mix and return the new droplet's uid."""
        uid = self.state.next_uid
        self.do(Mix(a, k, b, l))
        return uid

    def volume(self, uid: int) -> int:
        return self.state.droplet(uid).volume

    def discard_leftover(self, uid: int) -> None:
        """Send the droplet to waste if any of it is still on chip."""
        if self.state.has_droplet(uid):
            self.do(Discard(uid))

    def plan(self) -> tuple[Action, ...]:
        return tuple(self.actions)
