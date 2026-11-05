"""Published methods as "finish from here" planners: the leaf finisher of the search, in place
of the sharing pass's `finish_from`. MRCM (plan 4.2, paper log D-55) and the single-target
graph methods DFM, FacDA, Generic and DFMM (`GraphFinish`, D-60).

MRCM (`mrcm`) plans from an empty chip. From a mid-episode state the droplets already on chip
are offered the way e-MRCM offers the droplets an earlier target's tree made (`emrcm`): a child
equal to one of them, with enough of it, costs no reactant and no mix, and a target equal to one
is taken from it directly. On an empty chip this is MRCM's own plan. Targets go largest reduced
denominator first (e-MRCM's order); every remaining target needs demand 1 (MRCM's rule).
Only exact compositions are reused: a droplet left part-way through an n-input mix (built as a
chain of 2-input mixes) is one MRCM never names, so the finish rebuilds what it holds.

The trees are built through the sharing pass's builder (`share._Builder`), so leftovers wait in
a pool and are discarded only when a mix needs the room, and the cost is exact, final leftovers
not charged (as `finish_from`). The mixing models of a (composition, volume) do not depend on
the chip, so one `MrcmFinish` keeps them across calls: same trees, far less time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from meda.core.actions import Action, Output, Source
from meda.core.composition import ZERO, Composition, Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError
from meda.core.state import PrepState
from meda.solvers.baselines.emrcm import _overdrawn, _Shared, _SharedSearch
from meda.solvers.baselines.mrcm import _Tree, smooth
from meda.solvers.baselines.paper_graph import NotApplicable, PaperGraph, _fold, single_target
from meda.solvers.baselines.share import RELAXED_STEP_LIMIT, _Builder


class _CachedSearch(_SharedSearch):
    def __init__(self, *args: Any, cache: dict[Any, list[Any]]) -> None:
        super().__init__(*args)
        self.cache = cache

    def models(self, c: Composition, v: int):
        key = (c, v)
        if key not in self.cache:
            self.cache[key] = list(super().models(c, v))
        return self.cache[key]


class MrcmFinish:
    """`finish(state)`: MRCM's completion from `state` and its cost, or None when MRCM cannot
    finish from there (`fails` counts why, by the first words of the reason)."""

    def __init__(self, instance: Instance, weights: CostWeights) -> None:
        self.instance = instance
        self.relaxed = replace(instance, step_limit=RELAXED_STEP_LIMIT)
        self.weights = weights
        self.w = tuple(weights.fluid_weight(f.kind) for f in instance.fluids)
        self.lo = max(instance.chip.min_mix_volume, 2)
        self.m = instance.chip.max_mix_volume
        self.cache: dict[Any, list[Any]] = {}
        self.fails: dict[str, int] = {}

    def applicable(self) -> str | None:
        """Why MRCM cannot take this problem at all, or None."""
        for t in self.instance.targets:
            if t.demand != 1:
                return "MRCM makes one droplet of each target (demand 1)"
            if not smooth(t.composition.den, self.m):
                return f"{t.composition}: denominator has a prime factor above M = {self.m}"
        return None

    def __call__(self, state: PrepState) -> tuple[tuple[Action, ...], Rational] | None:
        try:
            return self.finish(state)
        except (NotApplicable, IllegalActionError) as e:
            why = " ".join(str(e).split()[:4])
            self.fails[why] = self.fails.get(why, 0) + 1
            return None

    def finish(self, state: PrepState) -> tuple[tuple[Action, ...], Rational]:
        if (why := self.applicable()) is not None:
            raise NotApplicable(why)
        b = _Builder(self.relaxed, state, [], self.weights, {d.uid for d in state.droplets})
        targets = self.instance.targets
        open_ids = [i for i in range(len(targets)) if state.remaining_demand[i]]
        for i in sorted(open_ids, key=lambda i: (-targets[i].composition.den, i)):
            target = targets[i]
            spare = self._spare(b)
            have = spare.get(target.composition)
            if have is not None and have[1] >= target.droplet_volume:
                root = have[0]
            else:
                offered = dict(spare)
                while True:
                    tree = _CachedSearch(
                        self.w, self.lo, self.m, offered, False, cache=self.cache
                    ).best(target.composition, target.droplet_volume)
                    if tree is None:
                        raise NotApplicable(f"MRCM finds no tree for {target.composition}")
                    over = _overdrawn(tree, offered)
                    if over is None:
                        break
                    del offered[over]
                b.pool -= {c.node for c in _shared(tree)}
                root = self._build(b, tree)
            b.pool.discard(root)
            b.do(Output(Source.droplet(root), target.id))
            b.discard_leftover(root)
            b.pool |= {d.uid for d in b.state.droplets}
        if not b.state.is_success:
            raise IllegalActionError(f"instance '{self.instance.instance_id}': demand left")
        return tuple(b.actions), b.cost

    @staticmethod
    def _spare(b: _Builder) -> dict[Composition, tuple[int, int]]:
        return MrcmFinish._spare_of(b.state)

    @staticmethod
    def _spare_of(state: PrepState) -> dict[Composition, tuple[int, int]]:
        """The largest droplet on chip of each composition: (uid, volume)."""
        out: dict[Composition, tuple[int, int]] = {}
        for d in sorted(state.droplets, key=lambda d: (d.volume, -d.uid)):
            out[d.composition] = (d.uid, d.volume)
        return out

    def _build(self, b: _Builder, tree: _Tree) -> int:
        """Build `tree` bottom up (children first, as MRCM's graph is emitted); its droplet's
        uid. A child's leftover joins the pool once its parent has taken from it."""
        n = self.instance.num_fluids
        inputs = [
            (Source.reservoir(f), u, Composition.pure(f, n)) for f, u in enumerate(tree.pure) if u
        ]
        kids = []
        for units, child in tree.children:
            uid = child.node if isinstance(child, _Shared) else self._build(b, child)
            kids.append(uid)
            inputs.append((Source.droplet(uid), units, b.state.droplet(uid).composition))
        uid = _fold(b, inputs, self.instance, len(b.actions))
        for k in kids:
            b.discard_leftover(k)
        return uid


def _shared(tree: _Tree) -> list[_Shared]:
    out: list[_Shared] = []
    stack = [tree]
    while stack:
        t = stack.pop()
        for _, child in t.children:
            if isinstance(child, _Shared):
                out.append(child)
            else:
                stack.append(child)
    return out


class GraphFinish:
    """`finish(state)`: a single-target published method (DFM, FacDA, Generic, DFMM) finishing
    from `state`, and its cost, or None (`fails` counts why). D-60.

    The method's graph is its own for the target from an empty chip (it does not depend on the
    chip), computed once. From a mid-episode state, a node whose composition is on chip with
    enough volume for the nodes that take from it is taken from that droplet instead of built,
    with everything only it needed; one droplet stands in for at most one node. The rest is
    built as `paper_graph.emit` builds it (each n-input node a chain of 2-input mixes), through
    the sharing pass's builder: leftovers wait in a pool, are discarded only when a mix needs the
    room, and the cost is exact, final leftovers not charged. On an empty chip it is the
    method's own plan. Like `MrcmFinish`, the method never chooses its graph around what is on
    chip; only exact compositions are reused."""

    def __init__(
        self,
        name: str,
        method: Callable[[Instance], PaperGraph],
        instance: Instance,
        weights: CostWeights,
    ) -> None:
        self.name = name
        self.instance = instance
        self.relaxed = replace(instance, step_limit=RELAXED_STEP_LIMIT)
        self.weights = weights
        self.fails: dict[str, int] = {}
        self.graph: PaperGraph | None = None
        self.why: str | None = None
        try:
            single_target(instance)
            self.graph = method(instance)
            self.comps = self.graph.compositions()
        except NotApplicable as e:
            self.why = str(e)

    def applicable(self) -> str | None:
        """Why the method cannot take this problem at all, or None."""
        return self.why

    def __call__(self, state: PrepState) -> tuple[tuple[Action, ...], Rational] | None:
        try:
            return self.finish(state)
        except (NotApplicable, IllegalActionError) as e:
            why = " ".join(str(e).split()[:4])
            self.fails[why] = self.fails.get(why, 0) + 1
            return None

    def finish(self, state: PrepState) -> tuple[tuple[Action, ...], Rational]:
        if self.graph is None:
            raise NotApplicable(self.why or f"{self.name} does not take this problem")
        if state.is_success:
            return (), ZERO
        g, comps = self.graph, self.comps
        target = self.instance.targets[0]
        root = len(g.nodes) - 1
        need = [0] * len(g.nodes)
        need[root] = target.droplet_volume
        spare = MrcmFinish._spare_of(state)
        given: dict[int, int] = {}  # node -> uid of the droplet on chip that stands in for it
        build = [False] * len(g.nodes)
        for i in range(root, -1, -1):  # parents before children (nodes are topological)
            if not need[i]:
                continue
            have = spare.get(comps[i])
            if have is not None and have[1] >= need[i]:
                given[i] = have[0]
                del spare[comps[i]]
                continue
            build[i] = True
            for x in g.nodes[i]:
                if x.node is not None and x.units > 0:
                    need[x.node] += x.units
        users = [0] * len(g.nodes)
        for i, node in enumerate(g.nodes):
            if build[i]:
                for j in {x.node for x in node if x.node is not None and x.units > 0}:
                    users[j] += 1
        b = _Builder(self.relaxed, state, [], self.weights, {d.uid for d in state.droplets})
        b.pool -= set(given.values())
        uid = dict(given)
        n = self.instance.num_fluids
        for i, node in enumerate(g.nodes):
            if not build[i]:
                continue
            units: dict[Source, int] = {}
            comp: dict[Source, Composition] = {}
            for x in node:
                if x.units <= 0:
                    continue
                if x.node is None:
                    assert x.fluid is not None
                    src, c = Source.reservoir(x.fluid), Composition.pure(x.fluid, n)
                else:
                    src, c = Source.droplet(uid[x.node]), comps[x.node]
                units[src] = units.get(src, 0) + x.units
                comp[src] = c
            uid[i] = _fold(b, [(s, units[s], comp[s]) for s in units], self.instance, i)
            for j in {x.node for x in node if x.node is not None and x.units > 0}:
                users[j] -= 1
                if not users[j]:
                    b.discard_leftover(uid[j])
        b.do(Output(Source.droplet(uid[root]), target.id))
        b.discard_leftover(uid[root])
        if not b.state.is_success:
            raise IllegalActionError(f"instance '{self.instance.instance_id}': demand left")
        return tuple(b.actions), b.cost
