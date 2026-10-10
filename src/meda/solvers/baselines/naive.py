"""Naive baseline: every target droplet built on its own, with no sharing (SPEC 10.4).

Always-available reference plan and the feasibility witness the generators use (SPEC 8.1).
Two constructions, both exact (they hit the target composition even when tolerance > 0):

1. **Chain** (any number of fluids), when `composition.den <= max_mix_volume`: draw `num[f]` units
   of each fluid in turn, mixing the running droplet with the next reservoir. Final volume is
   `den`, and every intermediate volume is at most `den`, so the mix limit holds.
2. **Mixed radix** (two fluids only), when `den` factorises into factors within
   `[min_mix_volume, max_mix_volume]`: write n in mixed radix over those factors and build
   `s_i = (a_i + s_{i-1}) / e_i` with two 2-input mixes per digit (the generalisation of
   bit-scanning; with M = 2 it *is* bit-scanning).

Outside that scope (a mixture whose denominator exceeds M, or a denominator with a prime factor
above M) `naive_plan` returns None: not found, which is never a claim of infeasibility.
"""

from __future__ import annotations

import time

from meda.core.actions import Action, Output, Source
from meda.core.composition import Composition, mix
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG
from meda.core.problem import Instance, Target
from meda.core.rules import IllegalActionError, replay
from meda.solvers.base import PlanBuilder, SolveBudget, SolveResult


def factorise_within(n: int, lo: int, hi: int) -> tuple[int, ...] | None:
    """Factors of `n`, each in `[lo, hi]`, or None if no such factorisation is found.

    Prime factors are combined greedily (largest first) into factors of at most `hi`.
    """
    if n < 2:
        return None
    primes: list[int] = []
    rest, d = n, 2
    while d * d <= rest:
        while rest % d == 0:
            primes.append(d)
            rest //= d
        d += 1
    if rest > 1:
        primes.append(rest)
    if any(p > hi for p in primes):
        return None
    primes.sort(reverse=True)
    factors: list[int] = []
    for p in primes:
        for i, f in enumerate(factors):
            if f * p <= hi:
                factors[i] = f * p
                break
        else:
            factors.append(p)
    if any(f < lo for f in factors):
        return None
    return tuple(sorted(factors))


def mixed_radix_digits(n: int, radices: tuple[int, ...]) -> tuple[int, ...]:
    """Digits a_i with `n = sum_i a_i * prod(radices[:i])` and `0 <= a_i < radices[i]`."""
    digits = []
    rest = n
    for e in radices:
        digits.append(rest % e)
        rest //= e
    if rest:
        raise ValueError(f"{n} does not fit the mixed radix {radices}")
    return tuple(digits)


def _chain(b: PlanBuilder, comp: Composition) -> int | None:
    """Build `comp` by drawing each fluid in turn. Needs `comp.den <= max_mix_volume`."""
    chip = b.instance.chip
    if not chip.min_mix_volume <= comp.den <= chip.max_mix_volume:
        return None
    used = [f for f, n in enumerate(comp.num) if n > 0]
    if len(used) < 2:
        return None  # pure fluid: the caller outputs straight from the reservoir
    first, second = used[0], used[1]
    uid = b.mix(
        Source.reservoir(first), comp.num[first], Source.reservoir(second), comp.num[second]
    )
    for f in used[2:]:
        uid = b.mix(Source.droplet(uid), b.volume(uid), Source.reservoir(f), comp.num[f])
    return uid


def _mixed_radix(b: PlanBuilder, comp: Composition) -> int | None:
    """Build a two-fluid `comp` digit by digit. Needs a factorisation of `comp.den`."""
    chip = b.instance.chip
    if comp.num_fluids != 2:
        return None
    radices = factorise_within(comp.den, chip.min_mix_volume, chip.max_mix_volume)
    if radices is None:
        return None
    hi, lo = 0, 1  # fluid 0 is the one whose share is n/den
    digits = mixed_radix_digits(comp.num[hi], radices)

    uid: int | None = None  # None means "pure fluid `lo`", i.e. share 0
    current = Composition.pure(lo, 2)
    for a, e in zip(digits, radices, strict=True):
        if uid is None:
            # current is pure `lo`: one mix from the two reservoirs gives a/e.
            if a == 0:
                continue  # still pure `lo`
            nxt = mix(Composition.pure(hi, 2), a, current, e - a)
            uid = b.mix(Source.reservoir(hi), a, Source.reservoir(lo), e - a)
            current = nxt
            continue
        # s_i = (a * hi + 1 * current + (e - a - 1) * lo) / e, as two 2-input mixes:
        # first `helper` = a/(e-1) at volume e-1, then helper + 1 unit of current.
        nxt = mix(Composition.from_ints((a, e - 1 - a), e - 1), e - 1, current, 1)
        if nxt == current:
            continue  # this digit does not move the composition
        prev_uid = uid
        if a == 0:
            helper = Source.reservoir(lo)
        elif a == e - 1:
            helper = Source.reservoir(hi)
        else:
            helper = Source.droplet(b.mix(Source.reservoir(hi), a, Source.reservoir(lo), e - 1 - a))
        uid = b.mix(helper, e - 1, Source.droplet(prev_uid), 1)
        current = nxt
        b.discard_leftover(prev_uid)
    if uid is not None and current != comp:  # defensive: the construction is exact
        raise AssertionError(f"mixed radix built {current}, expected {comp}")
    return uid


def _deliver_one(b: PlanBuilder, target: Target) -> bool:
    """Build and output one droplet of `target`. False if this instance is out of scope."""
    comp = target.composition
    if comp.den == 1:  # pure fluid: output straight from its reservoir
        b.do(Output(Source.reservoir(comp.num.index(1)), target.id))
        return True
    uid = _chain(b, comp)
    if uid is None:
        uid = _mixed_radix(b, comp)
    if uid is None or b.volume(uid) < target.droplet_volume:
        return False
    b.do(Output(Source.droplet(uid), target.id))
    b.discard_leftover(uid)
    return True


def naive_plan(instance: Instance) -> tuple[Action, ...] | None:
    """A legal plan that meets every target's demand, or None if out of scope.

    Each droplet is built from the reservoirs on its own: no intermediate is ever shared.
    """
    b = PlanBuilder.start(instance)
    try:
        for target in instance.targets:
            for _ in range(target.demand):
                if not _deliver_one(b, target):
                    return None
    except IllegalActionError:
        return None
    if not b.state.is_success:
        return None
    return tuple(b.actions)


class NaiveSolver:
    """`Solver` that returns the naive plan (status `feasible`), or `infeasible` if out of
    scope — which only means *this* construction found nothing."""

    name = "naive"

    def __init__(self, weights: CostWeights) -> None:
        self.weights = weights

    def solve(self, instance: Instance, budget: SolveBudget | None = None) -> SolveResult:
        start = time.perf_counter()
        actions = naive_plan(instance)
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


def plan_length_upper_bound(instance: Instance) -> int:
    """Actions the naive plan needs, without building it (used to size `step_limit`).

    Per delivered droplet: at most 2 mixes per radix digit, 1 output and 1 discard.
    """
    chip = instance.chip
    per_droplet = []
    for t in instance.targets:
        comp = t.composition
        if comp.den == 1:
            per_droplet.append(t.demand)
            continue
        radices = factorise_within(comp.den, chip.min_mix_volume, chip.max_mix_volume)
        digits = len(radices) if radices else comp.den
        nonzero = sum(1 for n in comp.num if n > 0)
        per_droplet.append(t.demand * (2 * digits + nonzero + 2))
    return sum(per_droplet)
