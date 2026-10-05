"""`Solver` protocol, `SolveBudget`, `SolveResult` (SPEC 10.1).

Agents are evaluated through the same `SolveResult` (see `meda.agents.base.run_episode`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from meda.core.composition import Rational
from meda.core.costs import StepCost
from meda.core.dag import MixingDAG
from meda.core.problem import Instance

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

    `cost` is the exact scalar plan cost under the weights used (no truncation penalty)."""

    instance_id: str
    producer: str
    status: SolveStatus
    dag: MixingDAG | None
    totals: StepCost | None
    cost: Rational | None
    wall_time: float
    nodes_expanded: int = 0


class Solver(Protocol):
    name: str

    def solve(self, instance: Instance, budget: SolveBudget) -> SolveResult: ...
