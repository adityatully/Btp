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
from functools import cache
from typing import NamedTuple

from meda.core.actions import Action, Output, Source
from meda.core.chip import ChipLimits
from meda.core.composition import Composition, Rational
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


MAX_EXTRA_STEPS = 4
"""How many dilute-up or blend steps one recipe may stack. Measured: 2 already builds every
dilution n/D (D <= 64, no prime factor above M) at every volume up to M, for M = 3..10; 4
leaves room for mixtures and tight storage. A bound, so the search ends: the input of such a
step can be finer than its output."""

SEARCH_BUDGET = 20_000
"""Recipe expansions one query may spend. Under tight storage a target may have no recipe,
and proving that would mean trying every one; the budget stops the search and the target is
"not found" (never "infeasible"). It counts expansions, not seconds, and every query starts
with an empty memo, so the outcome is deterministic."""

# A recipe step: ("split", e, s, 0), ("up", s, fluid, l) or ("blend", s, k, move), where
# `move` = i * F + j moves one 1/D unit from fluid j to fluid i (see `blend_inputs`).
_Step = tuple[str, int, int, int]
_Key = tuple[Composition, int, int, int, bool]  # comp, need, cap, extra, blend allowed


class _Found(NamedTuple):
    mixes: int
    peak: int  # most droplets on chip while it is built, counting its own
    volume: int  # units its droplet holds; 0 for a reservoir
    step: _Step | None  # None for a reservoir

    @property
    def is_droplet(self) -> bool:
        return self.step is not None


class _OutOfBudget(Exception):
    pass


def dilute_up_input(comp: Composition, s: int, fluid: int, l: int) -> Composition | None:
    """The droplet x_a with `Mix(x_a k, fluid l) = comp`, k = s - l, or None if it would need a
    negative amount of some fluid: x_a = (s * comp - l * e_fluid) / k."""
    k = s - l
    num = [s * n for n in comp.num]
    num[fluid] -= l * comp.den
    if num[fluid] < 0:
        return None
    return Composition.from_ints(tuple(num), k * comp.den)


def blend_inputs(
    comp: Composition, s: int, k: int, move: int
) -> tuple[Composition, Composition] | None:
    """Two droplets a, b with `Mix(a k, b s - k) = comp`: a is comp with one 1/D unit moved
    from fluid j to fluid i (`move` = i * F + j), b = (s * comp - k * a) / (s - k). None if
    either would need a negative amount of a fluid, or a == b."""
    f = comp.num_fluids
    i, j = divmod(move, f)
    if i == j or comp.num[j] == 0:
        return None
    na = list(comp.num)
    na[i] += 1
    na[j] -= 1
    nb = [s * x - k * y for x, y in zip(comp.num, na, strict=True)]
    if min(nb) < 0:
        return None
    a = Composition.from_ints(tuple(na), comp.den)
    b = Composition.from_ints(tuple(nb), (s - k) * comp.den)
    return None if a == b else (a, b)


def _left(sub: _Found, take: int) -> int:
    """1 if taking `take` units of the sub-recipe's droplet leaves some on chip."""
    return int(sub.is_droplet and sub.volume > take)


class _Search:
    """One recipe query on one chip: a memo of finished sub-recipes and an expansion budget.

    `recipe` returns a recipe for a droplet of at least `need` units of `comp` that never holds
    more than `cap` droplets on chip, or None. The split comes first, so every target the split
    alone builds keeps its plan. If it does not fit (its top mix is a multiple of a factor of
    D, so above M / 2 it often cannot hold the volume; or it needs more storage), the fewest-mix
    recipe whose last step is
    - a **dilute-up**: one mix of s units (need <= s <= M), k units of a built droplet x_a with
      l = s - k of a pure fluid (1/9 at volume 8 on M = 8 is 1 unit of 8/9 with 7 of buffer).
      It holds at most two droplets, so it also serves tight storage; else
    - a **blend** of two built droplets a and b, themselves built without blends (a mixture at
      volume M, where neither input can be pure: 5/16, 5/16, 3/16, 3/16 at volume 3 on M = 3).
    Ties go to the smallest s.
    """

    def __init__(self, chip: ChipLimits) -> None:
        self.lo, self.hi = chip.min_mix_volume, chip.max_mix_volume
        self.max_den = chip.max_denominator
        self.memo: dict[_Key, _Found | None] = {}
        self.left = SEARCH_BUDGET

    def find(self, comp: Composition, need: int, cap: int) -> tuple[_Found, _Key] | None:
        """Iterative deepening over the extra steps, so a short recipe is found first."""
        try:
            for extra in range(MAX_EXTRA_STEPS + 1):
                key = (comp, need, cap, extra, True)
                found = self.recipe(*key)
                if found is not None:
                    return found, key
        except _OutOfBudget:
            pass
        return None

    def recipe(
        self, comp: Composition, need: int, cap: int, extra: int, blend: bool
    ) -> _Found | None:
        if comp.den == 1:
            return _Found(0, 0, 0, None)
        if cap < 1:
            return None
        key = (comp, need, cap, extra, blend)
        if key in self.memo:
            return self.memo[key]
        if self.left == 0:
            raise _OutOfBudget
        self.left -= 1
        found = self._expand(comp, need, cap, extra, blend)
        self.memo[key] = found
        return found

    def _split(self, comp: Composition, e: int, s: int, cap: int, extra: int) -> _Found | None:
        """The split recipe and its exact storage peak, mirroring `build`: each piece is built
        while the running droplet waits on chip, then the two mix and leftovers go to waste."""
        pieces = split_pieces(comp, e)
        mixes, peak = len(pieces) - 1, 0
        running: _Found | None = None
        running_left = 0
        for piece, count in pieces:
            take = count * s
            held = int(running is not None and running.is_droplet)
            sub = self.recipe(piece, take, cap - held, extra, True)
            if sub is None:
                return None
            mixes += sub.mixes
            peak = max(peak, held + sub.peak)
            if running is None:
                running, running_left = sub, _left(sub, take)
                continue
            peak = max(peak, running_left + _left(sub, take) + 1)
            running = _Found(0, 0, 0, ("split", e, s, 0))  # a mix result, used up next
            running_left = 0
        return _Found(mixes, peak, e * s, ("split", e, s, 0))

    def _expand(
        self, comp: Composition, need: int, cap: int, extra: int, blend: bool
    ) -> _Found | None:
        top = _top_factor(comp, need, self.lo, self.hi)
        if top is not None:
            found = self._split(comp, top[0], top[1], cap, extra)
            if found is not None and found.peak <= cap:
                return found
        if extra == 0:
            return None
        best: _Found | None = None
        for s in range(max(need, self.lo), self.hi + 1):
            for fluid, units in enumerate(comp.num):
                for l in range(1, min(s - 1, s * units // comp.den) + 1):
                    xa = dilute_up_input(comp, s, fluid, l)
                    if xa is None or xa.den > self.max_den:
                        continue
                    sub = self.recipe(xa, s - l, cap, extra - 1, blend)
                    if sub is None:
                        continue
                    peak = max(sub.peak, _left(sub, s - l) + 1)
                    if peak <= cap and (best is None or sub.mixes + 1 < best.mixes):
                        best = _Found(sub.mixes + 1, peak, s, ("up", s, fluid, l))
        if best is not None or not blend:
            return best
        f = comp.num_fluids
        for s in range(max(need, self.lo), self.hi + 1):
            for k in range(1, s):
                for move in range(f * f):
                    pair = blend_inputs(comp, s, k, move)
                    if pair is None or max(pair[0].den, pair[1].den) > self.max_den:
                        continue
                    ra = self.recipe(pair[0], k, cap, extra - 1, False)
                    if ra is None:
                        continue
                    rb = self.recipe(pair[1], s - k, cap - int(ra.is_droplet), extra - 1, False)
                    if rb is None:
                        continue
                    mixes = ra.mixes + rb.mixes + 1
                    peak = max(
                        ra.peak, int(ra.is_droplet) + rb.peak, _left(ra, k) + _left(rb, s - k) + 1
                    )
                    if peak <= cap and (best is None or mixes < best.mixes):
                        best = _Found(mixes, peak, s, ("blend", s, k, move))
        return best


def recipe(chip: ChipLimits, comp: Composition, need: int) -> _Found | None:
    """The tree's recipe for `need` units of `comp` on `chip`, from an empty chip."""
    found = _Search(chip).find(comp, need, chip.storage_capacity)
    return None if found is None else found[0]


def build(b: PlanBuilder, comp: Composition, need: int) -> Source | None:
    """A source holding at least `need` units of `comp`, built from an empty chip: its
    reservoir if pure, else a new droplet. None if this construction cannot make it here."""
    if comp.den == 1:
        return Source.reservoir(comp.num.index(1))
    search = _Search(b.instance.chip)
    found = search.find(comp, need, b.instance.chip.storage_capacity)
    if found is None:
        return None
    return _emit(b, search, found[1])


def _emit(b: PlanBuilder, search: _Search, key: _Key) -> Source:
    """Emit the recipe the search found for `key` (every sub-recipe it used is in the memo)."""
    comp, need, cap, extra, blend = key
    if comp.den == 1:
        return Source.reservoir(comp.num.index(1))
    reused = b.reuse(comp, need)
    if reused is not None:
        return reused
    found = search.memo[key]
    assert found is not None and found.step is not None
    kind, x, y, z = found.step
    if kind == "up":
        s, fluid, l = x, y, z
        xa = dilute_up_input(comp, s, fluid, l)
        assert xa is not None
        src = _emit(b, search, (xa, s - l, cap, extra - 1, blend))
        out = Source.droplet(b.mix(src, s - l, Source.reservoir(fluid), l))
        if not src.is_reservoir:
            b.discard_leftover(src.index)
        return out
    if kind == "blend":
        s, k, move = x, y, z
        pair = blend_inputs(comp, s, k, move)
        assert pair is not None
        sa = _emit(b, search, (pair[0], k, cap, extra - 1, False))
        sb = _emit(b, search, (pair[1], s - k, cap - int(not sa.is_reservoir), extra - 1, False))
        out = Source.droplet(b.mix(sa, k, sb, s - k))
        for used in (sa, sb):
            if not used.is_reservoir:
                b.discard_leftover(used.index)
        return out
    e, s = x, y
    running: Source | None = None
    weight = 0
    for piece, count in split_pieces(comp, e):
        take = count * s
        held = int(running is not None and not running.is_reservoir)
        src = _emit(b, search, (piece, take, cap - held, extra, True))
        if running is None:
            running, weight = src, take
            continue
        before = running
        running = Source.droplet(b.mix(running, weight, src, take))
        weight += take
        for used in (before, src):
            if not used.is_reservoir:
                b.discard_leftover(used.index)
    assert running is not None
    return running


def within_tolerance(comp: Composition, tolerance: Rational, den: int) -> Composition | None:
    """A composition with denominator `den` within L-inf `tolerance` of `comp`, as close to it
    as rounding allows, or None if none exists. Exact: per fluid the allowed numerators are
    ceil((x - tol) den) .. floor((x + tol) den), clamped to [0, den], and they must be able to
    sum to `den`."""
    d, (p, q) = comp.den, (tolerance.num, tolerance.den)
    lo = [max(0, -((p * d - a * q) * den // (d * q))) for a in comp.num]  # ceil
    hi = [min(den, (a * q + p * d) * den // (d * q)) for a in comp.num]  # floor
    if any(x > y for x, y in zip(lo, hi, strict=True)) or not sum(lo) <= den <= sum(hi):
        return None
    # Start from the nearest numerators inside the bounds, then fix the sum one unit at a
    # time, always moving the fluid whose share is furthest from its target in that direction.
    num = [
        min(max((2 * a * den + d) // (2 * d), x), y)
        for a, x, y in zip(comp.num, lo, hi, strict=True)
    ]
    while sum(num) != den:
        up = sum(num) < den
        room = [i for i in range(len(num)) if (num[i] < hi[i] if up else num[i] > lo[i])]
        # error of fluid i in units of 1/(d * den): num_i * d - a_i * den
        i = min(room, key=lambda i: (num[i] * d - comp.num[i] * den) * (1 if up else -1))
        num[i] += 1 if up else -1
    return Composition.from_ints(tuple(num), den)


@cache
def _reachable_denominators(lo: int, hi: int, max_den: int) -> tuple[int, ...]:
    """1 and every denominator up to `max_den` that factorises into mix sizes in [lo, hi]."""
    return (1, *(d for d in range(2, max_den + 1) if factorise_within(d, lo, hi) is not None))


def aim(target: Target, chip: ChipLimits) -> Composition | None:
    """The composition the tree builds for `target`: the target itself when exact, else the
    one with the smallest denominator within the tolerance that M can reach (every prime
    factor <= M, at most `max_denominator`; FacDA's ratio transformation). A smaller
    denominator means a shallower recipe. None if no reachable composition is close enough."""
    comp, tol = target.composition, target.tolerance
    if tol.num == 0:
        return comp
    for den in _reachable_denominators(
        chip.min_mix_volume, chip.max_mix_volume, chip.max_denominator
    ):
        found = within_tolerance(comp, tol, den)
        if found is not None:
            return found
    return None


def min_denominator_within(comp: Composition, tolerance: Rational, cap: int) -> int:
    """The smallest denominator of any composition within `tolerance` of `comp` (reachable
    or not), searched up to `cap`; `comp.den` itself is always within. A lower bound on how
    fine a delivered droplet must be, for the step-limit depth bound."""
    if tolerance.num == 0:
        return comp.den
    for den in range(1, min(cap, comp.den) + 1):
        if within_tolerance(comp, tolerance, den) is not None:
            return den
    return comp.den


def _deliver(b: PlanBuilder, target: Target, count: int) -> bool:
    """Build one droplet for `count` deliveries of `target` and output them."""
    comp = aim(target, b.instance.chip)
    if comp is None:
        return False
    src = build(b, comp, count * target.droplet_volume)
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
    chip = b.instance.chip
    comp = aim(target, chip)
    if comp is None:
        return 1  # nothing to build; `_deliver` reports it
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
