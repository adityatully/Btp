"""Sharing pass: the constructions, with droplets kept on chip and reused across targets.

The naive and split-tree constructions (10.4) build every target from the reservoirs and send
every leftover to waste, so on many targets they pay for the same sample again and again
(E-5: 50-135% above the proved optimum). This pass builds the targets one delivery group at a
time on one chip and keeps every leftover in a **pool** of idle droplets. For each group it
prices, on a copy of the plan, every way it knows to make the droplet:

1. **direct**: a pool droplet within tolerance of the target;
2. **one mix**: a pool droplet with a reservoir or another pool droplet, k:l, k + l <= M;
3. **chain**: up to three mixes worked back from the target, each adding a fluid or an idle
   droplet, down to a pure fluid or a pool droplet (also a cheaper single-target recipe);
4. **tree**: the split-tree recipe from scratch, except that any piece of it already in the
   pool is taken from there (`PlanBuilder.reuse`);

and keeps the one with the least marginal cost per delivery (fluid, mixes, waste discarded on
the way). Leftovers stay on chip while storage allows; when a mix needs room, the idle droplet
furthest from every open target goes to waste first, so a recipe that fits an empty chip
always fits (the constructions' own storage accounting assumed those droplets were discarded).

Greedy choices depend on the order of the targets, so the pass runs a few orders (given,
dearest first, cheapest first, nearest neighbour) and returns the cheapest plan, and never one
dearer than `cheapest_construction`. `lookahead` = k adds a rollout over each group's k
cheapest candidates (pilot method). Every plan goes through `rules.apply`, so it is legal by
construction, step limit and storage included; a variant that breaks one is dropped.

**Finish from here** (`finish_from`, `SharingCostToFinish`; SPEC 10.3). The same pass started
from any state, with every droplet on chip in its pool: the cost of a real completion that
uses what has been built. The env's failure penalty uses it, so a half-built intermediate
lowers the penalty instead of being priced as waste (paper log E-18).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from functools import lru_cache

from meda.core.actions import Action, Discard, Mix, Output, Source
from meda.core.composition import ZERO, Composition, Rational, linf_distance, mix
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG
from meda.core.problem import Instance, Target
from meda.core.rules import IllegalActionError, apply, replay
from meda.core.state import PrepState
from meda.solvers.base import PlanBuilder, SolveBudget, SolveResult
from meda.solvers.baselines.tree import _deliveries_per_droplet, aim, build, dilute_up_input
from meda.solvers.fallback import (
    RELAXED_STEP_LIMIT,
    ConstructiveCostToFinish,
    cheapest_construction,
)


@dataclass(slots=True)
class _Builder(PlanBuilder):
    """A plan builder that keeps leftovers in a pool and discards them only when a mix needs
    the room. `cost` is the exact cost so far, final leftovers not counted."""

    weights: CostWeights = field(default=None)  # type: ignore[assignment]
    pool: set[int] = field(default_factory=set)
    cost: Rational = ZERO

    def copy(self) -> _Builder:
        return _Builder(
            self.instance, self.state, list(self.actions), self.weights, set(self.pool), self.cost
        )

    def do(self, action: Action) -> None:
        if isinstance(action, Mix):
            self._make_room(action)
        self._apply(action)

    def mix(self, a: Source, k: int, b: Source, l: int) -> int:
        """Apply the mix and return the new droplet's uid (taken after any discard that made
        room, since a waste node takes an id too)."""
        action = Mix(a, k, b, l)
        self._make_room(action)
        uid = self.state.next_uid
        self._apply(action)
        return uid

    def _apply(self, action: Action) -> None:
        self.state, step, _ = apply(self.state, action, self.instance)
        self.actions.append(action)
        self.cost = self.cost + self.weights.scalar(step, self.instance.fluids)
        self.pool = {uid for uid in self.pool if self.state.has_droplet(uid)}

    def _make_room(self, action: Mix) -> None:
        chip = self.instance.chip
        used = {s.index for s in (action.a, action.b) if not s.is_reservoir}
        if not all(self.state.has_droplet(uid) for uid in used):
            raise IllegalActionError(
                f"instance '{self.instance.instance_id}': a planned source is no longer on chip"
            )
        consumed = sum(
            1
            for s, take in ((action.a, action.k), (action.b, action.l))
            if not s.is_reservoir and self.state.droplet(s.index).volume == take
        )
        while len(self.state.droplets) - consumed + 1 > chip.storage_capacity:
            idle = [uid for uid in self.pool if uid not in used]
            if not idle:
                return  # apply reports the storage violation
            self._apply(Discard(max(idle, key=self._idleness)))

    def _idleness(self, uid: int) -> tuple[Rational, int, int]:
        """Larger = less useful: distance to the nearest open target, then smaller volume."""
        d = self.state.droplet(uid)
        open_targets = [
            t for t, r in zip(self.instance.targets, self.state.remaining_demand, strict=True) if r
        ]
        far = min(
            (linf_distance(d.composition, t.composition) for t in open_targets),
            default=Rational.of(1),
        )
        return far, -d.volume, uid

    def discard_leftover(self, uid: int) -> None:
        if self.state.has_droplet(uid):
            self.pool.add(uid)

    def reuse(self, composition: Composition, need: int) -> Source | None:
        fits = [
            uid
            for uid in self.pool
            if self.state.droplet(uid).composition == composition
            and self.state.droplet(uid).volume >= need
        ]
        if not fits:
            return None
        uid = min(fits, key=lambda u: (self.state.droplet(u).volume, u))
        self.pool.discard(uid)
        return Source.droplet(uid)

    def idle(self) -> list[tuple[int, Composition, int]]:
        return sorted(
            (uid, self.state.droplet(uid).composition, self.state.droplet(uid).volume)
            for uid in self.pool
        )


# A candidate emits the droplet for one delivery group and returns its source (or None).
_Emit = Callable[[_Builder], Source | None]


def _close(comp: Composition, target: Target) -> bool:
    return linf_distance(comp, target.composition) <= target.tolerance


def _one_mix(
    b: _Builder, target: Target, need: int
) -> Iterator[tuple[Rational, int, Source, int, Source, int]]:
    """Mixes of a pool droplet with a reservoir or another pool droplet that land within the
    target's tolerance and hold at least `need` units: (draw cost, volume, a, k, b, l)."""
    inst, chip = b.instance, b.instance.chip
    idle = b.idle()
    fluids = inst.fluids
    pure = [Composition.pure(f.id, inst.num_fluids) for f in fluids]
    partners: list[tuple[Source, Composition, int | None]] = [
        (Source.reservoir(f.id), pure[f.id], None) for f in fluids
    ] + [(Source.droplet(uid), c, v) for uid, c, v in idle]
    for i, (uid, ca, va) in enumerate(idle):
        for j, (src_b, cb, vb) in enumerate(partners):
            if not src_b.is_reservoir and j - len(fluids) <= i:
                continue  # each droplet pair once
            if cb == ca or not _pair_could_reach(ca, cb, target):
                continue
            for s in range(max(need, chip.min_mix_volume), chip.max_mix_volume + 1):
                for k in range(1, min(va, s - 1) + 1):
                    l = s - k
                    if vb is not None and l > vb:
                        continue
                    if not _take_ok(va, k, chip.min_droplet_volume) or (
                        vb is not None and not _take_ok(vb, l, chip.min_droplet_volume)
                    ):
                        continue
                    res = mix(ca, k, cb, l)
                    if res.den > chip.max_denominator or not _close(res, target):
                        continue
                    draw = (
                        b.weights.fluid_weight(fluids[src_b.index].kind) * l
                        if (src_b.is_reservoir)
                        else ZERO
                    )
                    yield draw, s, Source.droplet(uid), k, src_b, l


def _take_ok(volume: int, take: int, min_droplet: int) -> bool:
    return take == volume or volume - take >= min_droplet


def _pair_could_reach(ca: Composition, cb: Composition, target: Target) -> bool:
    """The target's share of each fluid must lie between the two inputs' shares, give or take
    the tolerance (a mix is a convex combination)."""
    t, tol = target.composition, target.tolerance
    for xa, xb, xt in zip(ca.num, cb.num, t.num, strict=True):
        lo = min(Rational.of(xa, ca.den), Rational.of(xb, cb.den))
        hi = max(Rational.of(xa, ca.den), Rational.of(xb, cb.den))
        share = Rational.of(xt, t.den)
        if share + tol < lo or share - tol > hi:
            return False
    return True


# A dilute-up chain: a base (a reservoir, or a pool droplet by uid), then steps
# (fluid, k, l): mix k units of the running droplet with l units of that fluid.
_Base = tuple[str, int]
_Chain = tuple[Rational, _Base, tuple[tuple[int, int, int], ...]]

CHAIN_DEPTH = 3
"""Most mixes in one chain. Measured on the E2 problems: 2 finds the tight-storage optima the
constructions miss (3/4 at volume 2 via 1/2); 3 adds little there and costs 10-50x the time.
On the full-size val sets 2 is 5-18x faster but 4-18% dearer per stage (paper log E-38)."""


BLEND_PARTNERS = 4
"""Idle droplets (nearest the target first) a chain may blend with. Each costs a branch at
every step; with a pool of 20-30 droplets the search took seconds per delivery group."""


def _chain(b: _Builder, aimed: Composition, need: int, depth: int) -> _Chain | None:
    """The cheapest dilute-up chain to `need` units of `aimed`: worked back from the last mix
    with `dilute_up_input`, down to a pure fluid or an idle pool droplet. Cost = fluid drawn +
    mixes (leftovers stay in the pool)."""
    inst, chip, w = b.instance, b.instance.chip, b.weights
    fluid_w = [w.fluid_weight(f.kind) for f in inst.fluids]
    idle: dict[Composition, tuple[int, int]] = {}
    for uid, comp, vol in b.idle():
        if comp not in idle or vol > idle[comp][1]:
            idle[comp] = (uid, vol)
    partners = sorted(idle.items(), key=lambda kv: (linf_distance(kv[0], aimed), kv[1][0]))
    partners = partners[:BLEND_PARTNERS]
    memo: dict[tuple[Composition, int, int], _Chain | None] = {}

    def search(x: Composition, take: int, left: int) -> _Chain | None:
        if x.den == 1:
            f = x.num.index(1)
            return fluid_w[f] * take, ("res", f), ()
        hit = idle.get(x)
        if hit is not None and hit[1] >= take and _take_ok(hit[1], take, chip.min_droplet_volume):
            return ZERO, ("pool", hit[0]), ()
        if left == 0:
            return None
        key = (x, take, left)
        if key in memo:
            return memo[key]
        best: _Chain | None = None
        last = left == 1  # an input at the last level ends the chain only as a fluid or in the pool
        for size in range(max(take, chip.min_mix_volume), chip.max_mix_volume + 1):
            for f, l, xp in _dilute_steps(x, size, chip.max_denominator):
                if last and xp.den != 1 and xp not in idle:
                    continue
                sub = search(xp, size - l, left - 1)
                if sub is not None:
                    cost = sub[0] + fluid_w[f] * l + w.w_mix
                    if best is None or cost < best[0]:
                        best = (cost, sub[1], (*sub[2], (f, size - l, l)))
            # the same step with an idle droplet in place of the fluid (a blend)
            for pc, (uid, pv) in partners:
                for l in range(1, min(size - 1, pv) + 1):
                    if not _take_ok(pv, l, chip.min_droplet_volume):
                        continue
                    xp = _unmix(x, size, pc, l)
                    if xp is None or xp == pc or xp.den > chip.max_denominator:
                        continue
                    if last and xp.den != 1 and xp not in idle:
                        continue
                    sub = search(xp, size - l, left - 1)
                    if sub is None or sub[1] == ("pool", uid):
                        continue
                    cost = sub[0] + w.w_mix
                    if best is None or cost < best[0]:
                        best = (cost, sub[1], (*sub[2], (-1 - uid, size - l, l)))
        memo[key] = best
        return best

    return search(aimed, need, depth)


@lru_cache(maxsize=1 << 14)
def _dilute_steps(
    x: Composition, size: int, max_den: int
) -> tuple[tuple[int, int, Composition], ...]:
    """(fluid, l, input) for every dilute-up step that ends at `x` in a `size`-unit mix, in the
    order the chain search tries them. Pure, so cached: every delivery group of a target works
    back from the same compositions, and rebuilding them was most of the pass's time."""
    out = []
    for f, units in enumerate(x.num):
        for l in range(1, min(size - 1, size * units // x.den) + 1):
            xp = dilute_up_input(x, size, f, l)
            if xp is not None and xp.den <= max_den:
                out.append((f, l, xp))
    return tuple(out)


@lru_cache(maxsize=1 << 16)
def _unmix(x: Composition, size: int, partner: Composition, l: int) -> Composition | None:
    """The droplet y with `Mix(y size - l, partner l) = x`, or None if it would need a
    negative amount of some fluid: y = (size * x - l * partner) / (size - l)."""
    num = [size * a * partner.den - l * p * x.den for a, p in zip(x.num, partner.num, strict=True)]
    if min(num) < 0:
        return None
    return Composition.from_ints(tuple(num), (size - l) * x.den * partner.den)


def _candidates(b: _Builder, target: Target, left: int) -> list[tuple[int, _Emit]]:
    """(deliveries served, emitter) for every way this pass knows to make the next droplet."""
    chip = b.instance.chip
    v = target.droplet_volume
    most = max(1, min(left, chip.max_mix_volume // v))
    out: list[tuple[int, _Emit]] = []
    # 1. direct
    for uid, comp, vol in b.idle():
        if vol >= v and _close(comp, target):
            out.append((min(left, vol // v), _fixed(Source.droplet(uid))))
    # 2. one mix: the cheapest draw per volume
    best: dict[int, tuple[Rational, Source, int, Source, int]] = {}
    for count in range(most, 0, -1):
        for draw, s, sa, k, sb, l in _one_mix(b, target, count * v):
            served = min(left, s // v)
            if served not in best or draw < best[served][0]:
                best[served] = (draw, sa, k, sb, l)
        if best:
            break
    for served, (_, sa, k, sb, l) in best.items():
        out.append((served, _mixer(sa, k, sb, l)))
    # 3. a dilute-up chain from a pure fluid or a pool droplet (exact aim)
    aimed = aim(target, chip)
    if aimed is not None and aimed.den > 1:
        for count in sorted({most, 1}, reverse=True):
            found = _chain(b, aimed, count * v, CHAIN_DEPTH)
            if found is not None:
                out.append((count, _chained(found[1], found[2])))
    # 4. tree from scratch, taking any piece already in the pool
    if aimed is not None:
        count = _deliveries_per_droplet(b, target, most)
        out.append((count, _tree(aimed, count * v)))
    return out


def _fixed(src: Source) -> _Emit:
    return lambda b: src


def _mixer(sa: Source, k: int, sb: Source, l: int) -> _Emit:
    def emit(b: _Builder) -> Source:
        uid = b.mix(sa, k, sb, l)
        for s in (sa, sb):
            if not s.is_reservoir:
                b.discard_leftover(s.index)
        return Source.droplet(uid)

    return emit


def _chained(base: _Base, steps: tuple[tuple[int, int, int], ...]) -> _Emit:
    def emit(b: _Builder) -> Source:
        kind, index = base
        src = Source.reservoir(index) if kind == "res" else Source.droplet(index)
        if kind == "pool":
            b.pool.discard(index)
        b.pool.difference_update(-1 - f for f, _, _ in steps if f < 0)  # held until used
        for f, k, l in steps:
            partner = Source.reservoir(f) if f >= 0 else Source.droplet(-1 - f)
            uid = b.mix(src, k, partner, l)
            for used in (src, partner):
                if not used.is_reservoir:
                    b.discard_leftover(used.index)
            src = Source.droplet(uid)
        return src

    return emit


def _tree(aimed: Composition, need: int) -> _Emit:
    return lambda b: build(b, aimed, need)


def _deliver(b: _Builder, target: Target, count: int, emit: _Emit) -> bool:
    src = emit(b)
    if src is None:
        return False
    if not src.is_reservoir:
        b.pool.discard(src.index)
    served = 0
    while served < count and b.state.remaining_demand[target.id]:
        b.do(Output(src, target.id))
        served += 1
    if not src.is_reservoir:
        b.discard_leftover(src.index)
    return served > 0


def _options(b: _Builder, tid: int) -> list[tuple[Rational, int, _Builder]]:
    """Every candidate for the next delivery group of target `tid`, applied to a copy of the
    plan: (marginal cost per delivery, deliveries served, the copy), cheapest first."""
    target = b.instance.targets[tid]
    left = b.state.remaining_demand[tid]
    out: list[tuple[Rational, int, _Builder]] = []
    for count, emit in _candidates(b, target, left):
        trial = b.copy()
        try:
            if not _deliver(trial, target, count, emit):
                continue
        except IllegalActionError:
            continue
        served = left - trial.state.remaining_demand[tid]
        out.append(((trial.cost - b.cost) * Rational.of(1, served), served, trial))
    out.sort(key=lambda o: (o[0], -o[1], len(o[2].actions)))
    return out


def _run(b: _Builder, order: list[int], lookahead: int) -> _Builder:
    """Finish the plan target by target in `order`. Each delivery group takes the cheapest
    candidate per delivery or, with `lookahead` = k > 0, the best of the k cheapest after each
    is finished greedily (a rollout). Raises IllegalActionError when no candidate is legal (a
    step limit, or storage no idle droplet can free)."""
    for i, tid in enumerate(order):
        while b.state.remaining_demand[tid]:
            options = _options(b, tid)
            if not options:
                raise IllegalActionError(
                    f"instance '{b.instance.instance_id}': no way to make target {tid}"
                )
            if lookahead == 0 or len(options) == 1:
                b = options[0][2]
                continue
            best: tuple[Rational, _Builder] | None = None
            for _, _, trial in options[:lookahead]:
                try:
                    done = _run(trial.copy(), order[i:], 0)
                except IllegalActionError:
                    continue
                if best is None or done.cost < best[0]:
                    best = (done.cost, trial)
            b = best[1] if best is not None else options[0][2]
    if not b.state.is_success:
        raise IllegalActionError(f"instance '{b.instance.instance_id}': demand left")
    return b


def _value(target: Target, instance: Instance, weights: CostWeights) -> Rational:
    """Fluid cost of one unit of the target: the dear ones are built first in one order."""
    c = target.composition
    return sum(
        (weights.fluid_weight(f.kind) * Rational.of(c.num[f.id], c.den) for f in instance.fluids),
        ZERO,
    )


def orders(instance: Instance, weights: CostWeights) -> list[list[int]]:
    """Target orders the pass tries: given, dearest first, cheapest first, nearest neighbour
    from the dearest."""
    ids = list(range(len(instance.targets)))
    value = {i: _value(instance.targets[i], instance, weights) for i in ids}
    dear = sorted(ids, key=lambda i: (-value[i], i))
    cheap = sorted(ids, key=lambda i: (value[i], i))
    chain = [dear[0]] if ids else []
    rest = set(ids) - set(chain)
    while rest:
        last = instance.targets[chain[-1]].composition
        nxt = min(rest, key=lambda i: (linf_distance(last, instance.targets[i].composition), i))
        chain.append(nxt)
        rest.discard(nxt)
    out: list[list[int]] = []
    for order in (ids, dear, cheap, chain):
        if order not in out:
            out.append(order)
    return out


def share_plan(
    instance: Instance, weights: CostWeights, lookahead: int = 0
) -> tuple[Action, ...] | None:
    """The cheapest plan over the target orders and the constructions, or None if neither
    finds one (not found, never a claim of infeasibility). `lookahead` = k rolls out the k
    cheapest candidates of every delivery group (about k x groups times slower)."""
    best: tuple[Rational, tuple[Action, ...]] | None = None
    base = cheapest_construction(instance, weights)
    if base is not None:
        best = (base[1], base[0])
    for order in orders(instance, weights):
        try:
            start = _Builder(instance, PrepState.initial(instance), [], weights)
            done = _run(start, order, lookahead)
        except IllegalActionError:
            continue
        if best is None or done.cost < best[0]:
            best = (done.cost, tuple(done.actions))
    return None if best is None else best[1]


def finish_from(
    state: PrepState, instance: Instance, weights: CostWeights
) -> tuple[tuple[Action, ...], Rational] | None:
    """The pass started from `state`, every droplet on chip in its pool: the cheapest
    completion over the target orders, with its cost (final leftovers not charged, as for a
    finished plan), or None if no order finishes. Run on a relaxed step limit: it answers
    "what would finishing cost", like the constructions' cost to finish (10.3)."""
    relaxed = replace(instance, step_limit=RELAXED_STEP_LIMIT)
    pool = {d.uid for d in state.droplets}
    best: tuple[tuple[Action, ...], Rational] | None = None
    for order in orders(instance, weights):
        open_order = [t for t in order if state.remaining_demand[t]]
        try:
            done = _run(_Builder(relaxed, state, [], weights, set(pool)), open_order, 0)
        except IllegalActionError:
            continue
        if best is None or done.cost < best[1]:
            best = (tuple(done.actions), done.cost)
    return best


class SharingCostToFinish:
    """`cost_to_finish(state, instance)` for `SamplePrepEnv` (SPEC 5.8, 10.3): the cheaper of
    `finish_from` and the constructions' cost to finish (waste everything, rebuild). Both
    are costs of real completions, so the minimum is one too. Picklable."""

    def __init__(self, weights: CostWeights) -> None:
        self.weights = weights
        self.construction = ConstructiveCostToFinish(weights)

    def __call__(self, state: PrepState, instance: Instance) -> Rational:
        rebuild = self.construction(state, instance)
        found = finish_from(state, instance, self.weights)
        return rebuild if found is None else min(rebuild, found[1])

    def supports(self, instance: Instance) -> bool:
        return self.construction.supports(instance)


class ShareSolver:
    """`Solver` for the sharing pass (status `feasible`), or `infeasible` if nothing was
    found, which only means this pass found nothing."""

    name = "share"

    def __init__(self, weights: CostWeights, lookahead: int = 0) -> None:
        self.weights = weights
        self.lookahead = lookahead

    def solve(self, instance: Instance, budget: SolveBudget | None = None) -> SolveResult:
        start = time.perf_counter()
        actions = share_plan(instance, self.weights, self.lookahead)
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
