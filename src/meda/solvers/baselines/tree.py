"""Split-tree baseline: builds any reachable target, over any number of fluids (SPEC 10.4).

**Reachability.** A mix of k + l units can only bring prime factors of k + l <= M into a
denominator, so a composition is reachable only if every prime factor of its (reduced)
denominator D is at most M. This construction builds every such composition, given
`min_mix_volume = 2`, enough storage for its depth, and `D <= max_denominator`.

**Construction.** Factor D into mix sizes e_1..e_k within `[min_mix_volume, M]`. At the top
level with factor e, the target is an equal-parts blend of e pieces, each with denominator
q = D / e: split the numerator vector into e consecutive chunks of q units (fluid by fluid, so
most chunks are pure). Identical chunks merge into one piece taken m times. Each distinct piece
is a reservoir (pure) or is built recursively; the pieces are combined pairwise into a running
droplet, `Mix(running, w, piece, m)`, with every mix at most e (times a volume scale s) <= M
units. Pieces are built and folded in one at a time, so the chip holds about two droplets per
level.

For example 3/16 S + 5/16 R + 1/2 B at M = 8 (16 = 2 x 8): pieces (3, 5, 0)/8 and pure B;
the first is `Mix(S 3, R 5)`, then `Mix(that 1, B 1)`.

The top level is scaled by s so its droplet holds the deliveries it serves (`s * e <= M`).
When no split fits the volume (often above M / 2, since a split's top mix is a multiple of a
factor of D), the droplet is made by a **dilute-up** instead: one mix of a built droplet with
a pure fluid, of any size up to M (`_recipe`). A target neither can build returns None.

Like naive, no droplet is shared between targets, so the plan is an upper bound on the
optimum, never a claim about it, and None means "not found", never "infeasible".
"""

from __future__ import annotations

import time
from functools import lru_cache

from meda.core.actions import Action, Output, Source
from meda.core.chip import ChipLimits
from meda.core.composition import Composition
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG
from meda.core.problem import Instance, Target
from meda.core.rules import IllegalActionError, replay
from meda.solvers.base import PlanBuilder, SolveBudget, SolveResult
from meda.solvers.baselines.naive import factorise_within


def split_pieces(comp: Composition, e: int) -> list[tuple[Composition, int]]:
    """`comp` as an equal-parts blend of `e` pieces: distinct pieces with multiplicities.

    Consecutive chunks of `den / e` units over the fluids in order, so the blend of the e
    chunks (each weight 1/e) is exactly `comp`.
    """
    q = comp.den // e
    chunks: list[list[int]] = []
    current = [0] * comp.num_fluids
    room = q
    for f, units in enumerate(comp.num):
        while units:
            take = min(units, room)
            current[f] += take
            units -= take
            room -= take
            if room == 0:
                chunks.append(current)
                current, room = [0] * comp.num_fluids, q
    counts: dict[Composition, int] = {}
    for chunk in chunks:
        piece = Composition.from_ints(tuple(chunk), q)
        counts[piece] = counts.get(piece, 0) + 1
    return list(counts.items())


def _top_factor(comp: Composition, need: int, lo: int, hi: int) -> tuple[int, int] | None:
    """(e, s): the top factor and volume scale with `need <= e * s <= hi`, least overshoot."""
    factors = factorise_within(comp.den, lo, hi)
    if factors is None:
        return None
    best: tuple[int, int, int] | None = None
    for e in sorted(set(factors), reverse=True):
        s = -(-need // e)  # ceil
        if e * s <= hi and (best is None or e * s - need < best[0]):
            best = (e * s - need, e, s)
    return None if best is None else (best[1], best[2])


MAX_DILUTE_UPS = 4
"""How many dilute-up steps one recipe may stack. Measured: 2 already builds every dilution
n/D (D <= 64, no prime factor above M) at every volume up to M, for M = 3..10; 4 leaves room
for mixtures. A bound, so the search ends: a dilute-up input can be finer than its output."""

# A recipe step: ("split", e, s, 0) or ("up", s, fluid, l).
_Step = tuple[str, int, int, int]


def dilute_up_input(comp: Composition, s: int, fluid: int, l: int) -> Composition | None:
    """The droplet x_a with `Mix(x_a k, fluid l) = comp`, k = s - l, or None if it would need a
    negative amount of some fluid: x_a = (s * comp - l * e_fluid) / k."""
    k = s - l
    num = [s * n for n in comp.num]
    num[fluid] -= l * comp.den
    if num[fluid] < 0:
        return None
    return Composition.from_ints(tuple(num), k * comp.den)


@lru_cache(maxsize=200_000)
def _recipe(
    comp: Composition, need: int, lo: int, hi: int, max_den: int, ups: int
) -> tuple[int, _Step | None] | None:
    """(mixes, first step) of a recipe for `need` units of `comp`, or None if none is found.

    The split comes first, so every target the split alone builds keeps its plan. Otherwise a
    **dilute-up**: the droplet is one mix of s units (need <= s <= M), k = s - l units of a
    built droplet x_a with l units of a pure fluid. That is what makes large droplets: a split
    top mix must be a multiple of a factor of D, so above M / 2 it often does not fit (1/9 at
    volume 8 on M = 8 is 1 unit of 8/9 with 7 of buffer). Among dilute-ups, the fewest mixes,
    then the smallest s.
    """
    if comp.den == 1:
        return 0, None
    top = _top_factor(comp, need, lo, hi)
    if top is not None:
        e, s = top
        total = len(split := split_pieces(comp, e)) - 1
        for piece, count in split:
            sub = _recipe(piece, count * s, lo, hi, max_den, ups)
            if sub is None:
                break
            total += sub[0]
        else:
            return total, ("split", e, s, 0)
    if ups == 0:
        return None
    best: tuple[int, _Step] | None = None
    for s in range(max(need, lo), hi + 1):
        for fluid, units in enumerate(comp.num):
            for l in range(1, min(s - 1, s * units // comp.den) + 1):
                xa = dilute_up_input(comp, s, fluid, l)
                if xa is None or xa.den > max_den:
                    continue
                sub = _recipe(xa, s - l, lo, hi, max_den, ups - 1)
                if sub is not None and (best is None or sub[0] + 1 < best[0]):
                    best = (sub[0] + 1, ("up", s, fluid, l))
    return best


def recipe(chip: ChipLimits, comp: Composition, need: int) -> tuple[int, _Step | None] | None:
    """(mixes, first step) of the tree's recipe for `need` units of `comp` on `chip`."""
    return _recipe(
        comp,
        need,
        chip.min_mix_volume,
        chip.max_mix_volume,
        chip.max_denominator,
        MAX_DILUTE_UPS,
    )


def build(b: PlanBuilder, comp: Composition, need: int, ups: int = MAX_DILUTE_UPS) -> Source | None:
    """A source holding at least `need` units of `comp`: its reservoir if pure, else a new
    droplet. None if this construction cannot make it on this chip."""
    if comp.den == 1:
        return Source.reservoir(comp.num.index(1))
    chip = b.instance.chip
    found = _recipe(comp, need, chip.min_mix_volume, chip.max_mix_volume, chip.max_denominator, ups)
    if found is None or found[1] is None:
        return None
    step = found[1]
    if step[0] == "up":
        _, s, fluid, l = step
        xa = dilute_up_input(comp, s, fluid, l)
        assert xa is not None
        src = build(b, xa, s - l, ups - 1)
        if src is None:
            return None
        out = Source.droplet(b.mix(src, s - l, Source.reservoir(fluid), l))
        if not src.is_reservoir:
            b.discard_leftover(src.index)
        return out
    _, e, s, _ = step
    running: Source | None = None
    weight = 0
    for piece, count in split_pieces(comp, e):
        take = count * s
        src = build(b, piece, take, ups)
        if src is None:
            return None
        if running is None:
            running, weight = src, take
            continue
        before = running
        running = Source.droplet(b.mix(running, weight, src, take))
        weight += take
        for used in (before, src):
            if not used.is_reservoir:
                b.discard_leftover(used.index)
    return running


def _deliver(b: PlanBuilder, target: Target, count: int) -> bool:
    """Build one droplet for `count` deliveries of `target` and output them."""
    src = build(b, target.composition, count * target.droplet_volume)
    if src is None:
        return False
    for _ in range(count):
        b.do(Output(src, target.id))
    if not src.is_reservoir:
        b.discard_leftover(src.index)
    return True


def _deliveries_per_droplet(b: PlanBuilder, target: Target, most: int) -> int:
    """The most deliveries (<= `most`) one droplet of `target` can serve: the largest whose
    total volume the split builds as one droplet, else the largest a dilute-up builds (32nds at
    M = 3 split only into mixes of 2, so a split droplet serves 2 unit deliveries; a dilute-up
    one serves 3). At least 1."""
    comp, chip = target.composition, b.instance.chip
    if comp.den == 1:
        return most  # a reservoir serves any number
    for count in range(most, 0, -1):
        need = count * target.droplet_volume
        if _top_factor(comp, need, chip.min_mix_volume, chip.max_mix_volume) is not None:
            return count
    for count in range(most, 0, -1):
        if recipe(chip, comp, count * target.droplet_volume) is not None:
            return count
    return 1


def tree_plan(instance: Instance) -> tuple[Action, ...] | None:
    """A legal plan that meets every target's demand, or None if out of scope.

    Each droplet serves as many deliveries of its target as fit in one mix (at most M units);
    nothing is shared between targets.
    """
    b = PlanBuilder.start(instance)
    m = instance.chip.max_mix_volume
    try:
        for target in instance.targets:
            left = target.demand
            per_droplet = max(1, m // target.droplet_volume)
            while left:
                count = _deliveries_per_droplet(b, target, min(left, per_droplet))
                if not _deliver(b, target, count):
                    return None
                left -= count
    except IllegalActionError:
        return None
    if not b.state.is_success:
        return None
    return tuple(b.actions)


class TreeSolver:
    """`Solver` that returns the split-tree plan (status `feasible`), or `infeasible` if out of
    scope, which only means this construction found nothing."""

    name = "tree"

    def __init__(self, weights: CostWeights) -> None:
        self.weights = weights

    def solve(self, instance: Instance, budget: SolveBudget | None = None) -> SolveResult:
        start = time.perf_counter()
        actions = tree_plan(instance)
        if actions is None:
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
        return SolveResult(
            instance_id=instance.instance_id,
            producer=self.name,
            status="feasible",
            dag=MixingDAG.from_deltas(instance, deltas, totals, self.name),
            totals=totals,
            cost=self.weights.scalar(totals, instance.fluids),
            wall_time=time.perf_counter() - start,
        )
