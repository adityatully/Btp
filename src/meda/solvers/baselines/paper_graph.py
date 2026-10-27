"""Mixing graphs as the papers draw them, and their replay through the engine (plan 2.4).

The published sample-preparation methods (DFM, FacDA, DFMM, MRCM, Generic; SPEC 10.4) output a
mixing graph whose nodes are n-input mixes: some units of each pure fluid plus some units of
earlier nodes. `PaperGraph` holds that graph as the paper states it, so a test can check the
paper's own numbers (mixes nm, units drawn, waste nw) on it. `emit` turns it into engine actions:
each n-input node becomes a chain of 2-input mixes (paper log D-29), every step goes through
`core.rules.apply`, so the plan is legal by construction and its DAG validates (rule 7). Our mix
count is then at least the paper's nm; both are reported.

A method that cannot take a problem raises `NotApplicable` with the reason (rule 9): never an
approximate plan.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass

from meda.core.actions import Action, Output, Source
from meda.core.composition import Composition, Rational, mix
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG
from meda.core.fluids import FluidKind
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.solvers.base import PlanBuilder, SolveBudget, SolveResult


class NotApplicable(Exception):
    """The method does not take this problem; the message says why."""


@dataclass(frozen=True, slots=True)
class In:
    """`units` of a pure fluid (`node` is None, `fluid` its id) or of an earlier node."""

    units: int
    fluid: int | None = None
    node: int | None = None


@dataclass(frozen=True, slots=True)
class PaperGraph:
    """Nodes in topological order; the last is the root, which holds the target."""

    num_fluids: int
    nodes: tuple[tuple[In, ...], ...]

    def volume(self, i: int) -> int:
        return sum(x.units for x in self.nodes[i])

    def compositions(self) -> list[Composition]:
        out: list[Composition] = []
        for i, node in enumerate(self.nodes):
            comp: Composition | None = None
            vol = 0
            for x in node:
                c = (
                    Composition.pure(x.fluid, self.num_fluids)
                    if x.node is None and x.fluid is not None
                    else out[_index(x, i)]
                )
                comp = c if comp is None else mix(comp, vol, c, x.units)
                vol += x.units
            if comp is None:
                raise ValueError(f"node {i} has no inputs")
            out.append(comp)
        return out

    def used(self) -> list[bool]:
        """Nodes the root depends on (a node no one takes from is never built)."""
        keep = [False] * len(self.nodes)
        keep[-1] = True
        for i in range(len(self.nodes) - 1, -1, -1):
            if keep[i]:
                for x in self.nodes[i]:
                    if x.node is not None and x.units > 0:
                        keep[x.node] = True
        return keep

    def counts(self, delivered: int) -> dict[str, object]:
        """The paper's own metrics: n-input mixes, units drawn per fluid, waste."""
        keep = self.used()
        drawn = [0] * self.num_fluids
        taken = [0] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if not keep[i]:
                continue
            for x in node:
                if x.node is None and x.fluid is not None:
                    drawn[x.fluid] += x.units
                elif x.node is not None:
                    taken[x.node] += x.units
        made = sum(self.volume(i) for i in range(len(self.nodes)) if keep[i])
        return {
            "nm": sum(keep),
            "drawn": tuple(drawn),
            "nw": made - sum(taken) - delivered,
        }


def _index(x: In, i: int) -> int:
    if x.node is None or not 0 <= x.node < i:
        raise ValueError(f"node {i} takes from {x.node}, which is not an earlier node")
    return x.node


def sample_and_buffer(instance: Instance) -> tuple[int, int]:
    """(sample, buffer) fluid ids of a two-fluid instance."""
    if instance.num_fluids != 2:
        raise NotApplicable(f"{instance.num_fluids} fluids: a dilution method takes 2")
    kinds = [f.kind for f in instance.fluids]
    buffer = kinds.index(FluidKind.BUFFER) if FluidKind.BUFFER in kinds else 1
    return 1 - buffer, buffer


def single_target(instance: Instance) -> None:
    if len(instance.targets) != 1 or instance.targets[0].demand != 1:
        raise NotApplicable("the method makes one droplet of one target")


def emit(instance: Instance, graph: PaperGraph) -> tuple[Action, ...]:
    """Engine actions that build `graph` and output its root to the instance's one target.

    Inputs of a node are merged by source and folded into a running droplet one at a time, in
    the first order whose every intermediate mix is legal (a result equal to one of its sources,
    or a denominator above the chip's limit, is not). Leftovers stay on chip and go to waste at
    the end (storage is loose on stages 1-2). A plan longer than the instance's step limit,
    which binds every producer, is not applicable."""
    single_target(instance)
    target = instance.targets[0]
    if graph.num_fluids != instance.num_fluids:
        raise ValueError("graph and instance disagree on the number of fluids")
    if graph.volume(len(graph.nodes) - 1) < target.droplet_volume:
        raise NotApplicable(
            f"the method's last mix makes {graph.volume(len(graph.nodes) - 1)} units, "
            f"the target droplet is {target.droplet_volume}"
        )
    comps = graph.compositions()
    keep = graph.used()
    steps = 1 + sum(
        len({(x.fluid, x.node) for x in node if x.units}) - 1
        for i, node in enumerate(graph.nodes)
        if keep[i]
    )
    if steps > instance.step_limit:
        raise NotApplicable(
            f"the plan needs {steps} steps, the step limit is {instance.step_limit}"
        )
    b = PlanBuilder.start(instance)
    uid: dict[int, int] = {}
    for i, node in enumerate(graph.nodes):
        if keep[i]:
            uid[i] = _fold(b, _merged(node, uid, comps), instance, i)
    b.do(Output(Source.droplet(uid[len(graph.nodes) - 1]), target.id))
    return tuple(b.actions)


def _merged(
    node: tuple[In, ...], uid: dict[int, int], comps: list[Composition]
) -> list[tuple[Source, int, Composition]]:
    units: dict[Source, int] = {}
    comp: dict[Source, Composition] = {}
    for x in node:
        if x.units <= 0:
            continue
        if x.node is None and x.fluid is not None:
            src = Source.reservoir(x.fluid)
            comp[src] = Composition.pure(x.fluid, comps[0].num_fluids)
        else:
            assert x.node is not None
            src = Source.droplet(uid[x.node])
            comp[src] = comps[x.node]
        units[src] = units.get(src, 0) + x.units
    return [(s, units[s], comp[s]) for s in units]


def _fold(
    b: PlanBuilder, inputs: list[tuple[Source, int, Composition]], instance: Instance, i: int
) -> int:
    if len({c for _, _, c in inputs}) < 2:
        raise NotApplicable(f"node {i} mixes only one composition, which is not a mix")
    limit = instance.chip.max_denominator
    for order in itertools.permutations(inputs):
        comp, vol, ok = order[0][2], order[0][1], True
        for _, units, c in order[1:]:
            nxt = mix(comp, vol, c, units)
            if nxt in (comp, c) or nxt.den > limit:
                ok = False
                break
            comp, vol = nxt, vol + units
        if ok:
            break
    else:
        raise NotApplicable(f"node {i} has no legal order of 2-input mixes")
    (a, ka, _), (c2, kc, _) = order[0], order[1]
    run = b.mix(a, ka, c2, kc)
    for src, units, _ in order[2:]:
        run = b.mix(Source.droplet(run), b.volume(run), src, units)
    return run


def plan_of(method: Callable[[Instance], PaperGraph], instance: Instance) -> tuple[Action, ...]:
    return emit(instance, method(instance))


class PaperSolver:
    """`Solver` around a paper's graph builder: `feasible` with the replayed plan, or
    `infeasible` when the method does not take the problem (only: *this* method)."""

    verified = True

    def __init__(
        self, name: str, method: Callable[[Instance], PaperGraph], weights: CostWeights
    ) -> None:
        self.name = name
        self.method = method
        self.weights = weights

    def solve(self, instance: Instance, budget: SolveBudget | None = None) -> SolveResult:
        start = time.perf_counter()
        try:
            actions = plan_of(self.method, instance)
        except NotApplicable:
            return SolveResult(
                instance.instance_id,
                self.name,
                "infeasible",
                None,
                None,
                None,
                time.perf_counter() - start,
            )
        _, totals, deltas = replay(instance, actions)
        cost: Rational = self.weights.scalar(totals, instance.fluids)
        return SolveResult(
            instance_id=instance.instance_id,
            producer=self.name,
            status="feasible",
            dag=MixingDAG.from_deltas(instance, deltas, totals, self.name),
            totals=totals,
            cost=cost,
            wall_time=time.perf_counter() - start,
        )
