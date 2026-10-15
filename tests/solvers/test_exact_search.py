"""Exact search, cost-to-finish fallback, bitscan and the facda stub (SPEC 10; milestone M5).

M5's acceptance criteria are here: the hand-checked cases of SPEC 17.3 give their stated optima,
and the exact search proves and stores an optimum for every instance of `tiny_exact_v1`.
"""

from __future__ import annotations

from dataclasses import replace
from math import lcm

import pytest

from meda.core.actions import Mix, Output, Source
from meda.core.composition import Rational
from meda.core.dag import validate
from meda.core.problem import Instance
from meda.core.rules import apply, replay
from meda.core.state import PrepState
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.solvers.base import SolveBudget
from meda.solvers.baselines.bitscan import BitScanSolver, applies_to, is_dyadic
from meda.solvers.baselines.facda import FacDASolver
from meda.solvers.baselines.naive import NaiveSolver, naive_plan
from meda.solvers.exact_search import ExactSearch, fluid_bound_heuristic, zero_heuristic
from meda.solvers.fallback import NaiveCostToFinish, UnsupportedFallbackError
from tests.core.helpers import chip, default_weights, dilution, mixture

WEIGHTS = default_weights()
S, B = Source.reservoir(0), Source.reservoir(1)
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)
BUDGET = SolveBudget(max_seconds=60)


def at_accuracy(instance: Instance) -> Instance:
    """The instance at the accuracy level its targets need (as `tiny_exact_v1` is built)."""
    den = lcm(*[t.composition.den for t in instance.targets])
    return replace(instance, chip=replace(instance.chip, max_denominator=den))


def optimum(instance: Instance) -> tuple[Rational, int]:
    """(exact optimal cost, mix count) of a validated optimal plan."""
    solver = ExactSearch(WEIGHTS)
    result = solver.solve(instance, BUDGET)
    assert result.status == "optimal", f"{instance.instance_id}: {result.status}"
    assert result.dag is not None and result.cost is not None and result.totals is not None
    validate(result.dag, instance).raise_if_invalid()
    return result.cost, result.totals.mix_ops


def cost_of(instance: Instance, plan: list) -> Rational:
    _, totals, _ = replay(instance, plan)
    return WEIGHTS.scalar(totals, instance.fluids)


# --------------------------------------------------------------------------- SPEC 17.3


def test_case1_quarter_m2_optimum_is_two_mixes() -> None:
    inst = at_accuracy(dilution([(1, 4)], chip_limits=chip("dmfb_m2")))
    assert optimum(inst) == (Rational.of(15), 2)


def test_case2_quarter_m4_prefers_two_mixes_over_one() -> None:
    """SPEC 17.3 case 2 notes the answer depends on the weights, and it does.

    One mix suffices at M = 4 (S1 + B3 -> 1/4, volume 4) but costs 31/2, because it draws 3
    units of buffer and wastes 3. The 2-mix plan of case 1 is legal here too and costs 15, so
    that is the optimum under the default weights.
    """
    inst = at_accuracy(dilution([(1, 4)], chip_limits=chip("meda_m4")))
    one_mix = [Mix(S, 1, B, 3), Output(Source.droplet(2), 0)]
    assert cost_of(inst, one_mix) == Rational.of(31, 2)
    assert optimum(inst) == (Rational.of(15), 2)
    # Weighting waste and buffer at zero makes the single mix optimal instead.
    cheap_waste = replace(WEIGHTS, w_waste=Rational.of(0), w_buffer=Rational.of(0))
    result = ExactSearch(cheap_waste).solve(inst, BUDGET)
    assert result.totals is not None and result.totals.mix_ops == 1


def test_case3_three_tenths_m8() -> None:
    """A 2-mix plan exists, but the cost optimum uses 3 mixes and one unit less sample.

    SPEC 17.3 case 3 describes S2 + B3 -> 2/5, then 3 units of 2/5 + B1 -> 3/10: 2 mixes, and it
    costs 29. The optimum under the default weights is 41/2: S1 + B1 -> 1/2; 1 unit of 1/2 + B4
    -> 1/10; the other unit of 1/2 + 1 unit of 1/10 -> 3/10. That is 3 mixes but draws 1 unit of
    sample instead of 2, and sample costs 10 a unit against 1 for a mix. Minimising mixes and
    minimising cost are different objectives, and SPEC's cost weights put sample first.
    """
    inst = at_accuracy(dilution([(3, 10)]))
    spec_plan = [
        Mix(S, 2, B, 3),
        Mix(Source.droplet(2), 3, B, 1),
        Output(Source.droplet(3), 0),
    ]
    assert cost_of(inst, spec_plan) == Rational.of(29)
    cost, mixes = optimum(inst)
    assert (cost, mixes) == (Rational.of(41, 2), 3)
    _, totals, _ = replay(inst, list(ExactSearch(WEIGHTS).search(inst, BUDGET)[0] or ()))
    assert totals.reagent_by_fluid == (1, 5)  # one unit of sample, five of buffer
    # Charging mixes heavily instead recovers the 2-mix plan.
    mix_heavy = replace(WEIGHTS, w_mix=Rational.of(50))
    result = ExactSearch(mix_heavy).solve(inst, BUDGET)
    assert result.totals is not None and result.totals.mix_ops == 2


def test_case3_infeasible_on_a_1to1_chip() -> None:
    """3/10 is not dyadic, so 1:1 mixing cannot reach it (SPEC 17.3 case 3).

    Proving infeasibility means exhausting the space, not just failing to find a plan, so the
    instance is bounded tightly (8 steps, 3 droplets). Left at the chip's default 10 droplets
    and 30 steps the same search only reports `timeout` -- it cannot prove anything -- which is
    the honest answer and why the status distinguishes the two.
    """
    bounded = dilution(
        [(3, 10)],
        chip_limits=chip("dmfb_m2", max_denominator=10, storage_capacity=3),
        step_limit=8,
    )
    solver = ExactSearch(WEIGHTS)
    result = solver.solve(bounded, SolveBudget(max_seconds=30))
    assert result.status == "infeasible"
    assert result.dag is None
    assert solver.last_stats is not None and solver.last_stats.exhausted
    # The naive and bit-scanning baselines decline it outright.
    assert naive_plan(bounded) is None
    assert BitScanSolver(WEIGHTS).solve(bounded).status == "infeasible"


def test_case4_sharing_m2_optimum_is_three_mixes() -> None:
    """The headline case: sharing the 1/2 droplet beats building each target alone."""
    inst = at_accuracy(dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2")))
    cost, mixes = optimum(inst)
    assert mixes == 3
    assert cost == Rational.of(26)
    naive = NaiveSolver(WEIGHTS).solve(inst)
    assert naive.totals is not None and naive.totals.mix_ops == 4
    assert naive.cost is not None and cost < naive.cost


# --------------------------------------------------------------------------- M5 acceptance


@pytest.mark.parametrize("instance", TINY.instances, ids=lambda i: i.instance_id)
def test_every_tiny_instance_has_a_stored_optimum(instance: Instance) -> None:
    """Acceptance: the frozen optimum is reproduced, is optimal, and is a valid plan."""
    stored = TINY.reference[instance.instance_id]
    assert stored.status == "optimal"
    cost, mixes = optimum(instance)
    assert cost == stored.cost
    assert mixes == stored.detail["mix_ops"]


@pytest.mark.parametrize("instance", TINY.instances, ids=lambda i: i.instance_id)
def test_optimum_never_exceeds_the_naive_plan(instance: Instance) -> None:
    naive = NaiveSolver(WEIGHTS).solve(instance)
    assert naive.cost is not None
    assert TINY.reference[instance.instance_id].cost <= naive.cost


def test_tiny_set_runs_its_chips_at_a_matched_accuracy_level() -> None:
    """The property that makes the set exactly searchable, enforced so it cannot drift."""
    for inst in TINY.instances:
        assert inst.chip.max_denominator == lcm(*[t.composition.den for t in inst.targets])


# --------------------------------------------------------------------------- search internals


@pytest.mark.parametrize("instance", TINY.instances[:6], ids=lambda i: i.instance_id)
def test_heuristic_and_plain_dijkstra_agree(instance: Instance) -> None:
    """A* with the fluid bound must return the same cost as uniform-cost search."""
    with_h = ExactSearch(WEIGHTS, heuristic="fluid_bound").solve(instance, BUDGET)
    without = ExactSearch(WEIGHTS, heuristic="zero").solve(instance, BUDGET)
    assert with_h.status == without.status == "optimal"
    assert with_h.cost == without.cost


@pytest.mark.parametrize("instance", TINY.instances[:6], ids=lambda i: i.instance_id)
def test_branch_and_bound_does_not_change_the_answer(instance: Instance) -> None:
    bounded = ExactSearch(WEIGHTS, use_incumbent=True).solve(instance, BUDGET)
    plain = ExactSearch(WEIGHTS, use_incumbent=False).solve(instance, BUDGET)
    assert bounded.cost == plain.cost


def test_fluid_bound_never_exceeds_the_true_remaining_cost() -> None:
    """Admissibility, checked at every state along an optimal plan."""
    inst = at_accuracy(dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2")))
    heuristic = fluid_bound_heuristic(WEIGHTS)
    plan, total, _ = ExactSearch(WEIGHTS).search(inst, BUDGET)
    assert plan is not None and total is not None
    state = PrepState.initial(inst)
    spent = Rational.of(0)
    for action in plan:
        remaining = total - spent
        assert heuristic(state, inst) <= remaining
        assert zero_heuristic(state, inst) <= remaining
        state, step_cost, _ = apply(state, action, inst)
        spent = spent + WEIGHTS.scalar(step_cost, inst.fluids)
    assert spent == total


def test_budget_reports_timeout_not_a_wrong_answer() -> None:
    """A set-up the search cannot finish must say `timeout`, never invent an optimum."""
    hard = dilution([(7, 64)], step_limit=40)  # max_denominator 256, M = 8
    solver = ExactSearch(WEIGHTS, use_incumbent=False)
    result = solver.solve(hard, SolveBudget(max_nodes=200))
    assert result.status == "timeout"
    assert result.cost is None and result.dag is None
    assert solver.last_stats is not None and not solver.last_stats.exhausted


def test_incumbent_is_returned_when_nothing_cheaper_exists() -> None:
    """With the naive plan already optimal, the search proves it by exhausting the space."""
    inst = at_accuracy(dilution([(1, 8)], chip_limits=chip("dmfb_m2")))
    solver = ExactSearch(WEIGHTS)
    result = solver.solve(inst, BUDGET)
    naive = NaiveSolver(WEIGHTS).solve(inst)
    assert result.cost == naive.cost
    assert result.status == "optimal"


def test_search_is_deterministic() -> None:
    inst = at_accuracy(dilution([(5, 8)], chip_limits=chip("dmfb_m2")))
    first = ExactSearch(WEIGHTS).search(inst, BUDGET)
    second = ExactSearch(WEIGHTS).search(inst, BUDGET)
    assert first[0] == second[0]
    assert first[1] == second[1]


# --------------------------------------------------------------------------- fallback


@pytest.mark.parametrize("instance", TINY.instances, ids=lambda i: i.instance_id)
def test_cost_to_finish_is_an_upper_bound_at_the_start(instance: Instance) -> None:
    """From the initial state it must be at least the optimum, or truncating would look cheap."""
    fallback = NaiveCostToFinish(WEIGHTS)
    estimate = fallback(PrepState.initial(instance), instance)
    assert estimate >= TINY.reference[instance.instance_id].cost


def test_cost_to_finish_charges_what_is_on_chip_and_drops_to_zero_when_done() -> None:
    inst = dilution([(1, 4)], chip_limits=chip("dmfb_m2"))
    fallback = NaiveCostToFinish(WEIGHTS)
    start = PrepState.initial(inst)
    assert fallback(start, inst) == Rational.of(15)  # the naive plan's cost
    mid, _, _ = replay(inst, [Mix(S, 1, B, 1)])
    # Same demand, but two units of 1/2 are now on chip and would be wasted.
    assert fallback(mid, inst) == Rational.of(15) + Rational.of(2)
    done, _, _ = replay(
        inst, [Mix(S, 1, B, 1), Mix(Source.droplet(2), 1, B, 1), Output(Source.droplet(3), 0)]
    )
    assert done.is_success
    assert fallback(done, inst) == Rational.of(0)


def test_cost_to_finish_scales_with_remaining_demand_and_caches() -> None:
    inst = dilution([(1, 4)], chip_limits=chip("dmfb_m2"), demand=3, step_limit=60)
    fallback = NaiveCostToFinish(WEIGHTS)
    full = fallback.remaining_cost((3,), inst)
    assert fallback.remaining_cost((1,), inst) * 3 == full
    assert fallback.remaining_cost((0,), inst) == Rational.of(0)
    assert len(fallback._cache) == 2  # (3,) and (1,); the empty demand short-circuits
    assert fallback.remaining_cost((3,), inst) == full  # served from the cache


def test_cost_to_finish_raises_for_an_unsupported_mixture() -> None:
    """Rule 9: name the instance and the limit instead of inventing a penalty."""
    hard = mixture([(5, 3, 2)])  # denominator 10 > M for a 3-fluid chain
    fallback = NaiveCostToFinish(WEIGHTS)
    assert not fallback.supports(hard)
    with pytest.raises(UnsupportedFallbackError, match=r"(?s)test-mixture.*cost-to-finish"):
        fallback(PrepState.initial(hard), hard)
    assert fallback.supports(dilution([(1, 4)], chip_limits=chip("dmfb_m2")))


# --------------------------------------------------------------------------- bitscan, facda


def test_bitscan_matches_naive_on_dyadic_dilutions() -> None:
    """Under M = 2 the naive mixed-radix construction is bit-scanning; they must agree."""
    for numerator in range(1, 8):
        inst = dilution([(numerator, 8)], chip_limits=chip("dmfb_m2"), step_limit=60)
        scan = BitScanSolver(WEIGHTS).solve(inst)
        naive = NaiveSolver(WEIGHTS).solve(inst)
        assert scan.status == "feasible"
        assert scan.cost == naive.cost
        assert scan.totals == naive.totals
        assert scan.dag is not None
        validate(scan.dag, inst).raise_if_invalid()
        assert scan.dag.producer == "bitscan"


@pytest.mark.parametrize(
    ("instance", "reason"),
    [
        (dilution([(1, 4)]), "not 1:1"),  # meda_m8
        (mixture([(1, 1, 1)], chip_limits=chip("dmfb_m2")), "3 fluids"),
        (dilution([(1, 3)], chip_limits=chip("dmfb_m2")), "not a dyadic"),
    ],
)
def test_bitscan_declines_outside_its_scope(instance: Instance, reason: str) -> None:
    assert reason in (applies_to(instance) or "")
    assert BitScanSolver(WEIGHTS).solve(instance).status == "infeasible"


def test_is_dyadic() -> None:
    assert [n for n in range(1, 17) if is_dyadic(n)] == [1, 2, 4, 8, 16]


def test_facda_is_an_unverified_stub() -> None:
    solver = FacDASolver(WEIGHTS)
    assert solver.verified is False
    with pytest.raises(NotImplementedError, match="reproduce the paper"):
        solver.solve(dilution([(1, 4)]))


def test_naive_is_never_cheaper_than_the_optimum() -> None:
    """Sanity: the baseline cannot beat a proved optimum on any tiny instance."""
    for inst in TINY.instances:
        plan = naive_plan(inst)
        assert plan is not None
        assert cost_of(inst, list(plan)) >= TINY.reference[inst.instance_id].cost


@pytest.mark.parametrize("max_nodes", [0, 1, 5, 25, 100])
def test_timeout_lower_bound_never_exceeds_a_proved_optimum(max_nodes: int) -> None:
    """Stop the search early on every solved instance: the bound must sit at or below the
    stored optimum, and any best-known plan must be real and no cheaper than it."""
    for inst in TINY.instances:
        optimum = TINY.reference[inst.instance_id].cost
        result = ExactSearch(WEIGHTS).solve(inst, SolveBudget(max_nodes=max_nodes))
        if result.status == "optimal":  # finished within the budget
            assert result.cost == optimum
            continue
        assert result.status == "timeout"
        assert result.lower_bound is not None
        assert result.lower_bound <= optimum, inst.instance_id
        if result.dag is not None:
            assert result.cost is not None and result.cost >= optimum
            assert result.lower_bound <= result.cost
            validate(result.dag, inst).raise_if_invalid()


def test_bound_before_any_expansion_is_the_root_heuristic() -> None:
    inst = at_accuracy(dilution([(7, 32)], chip_limits=chip("dmfb_m2")))
    result = ExactSearch(WEIGHTS, use_incumbent=False).solve(inst, SolveBudget(max_nodes=0))
    root = PrepState.initial(inst)
    assert result.status == "timeout" and result.dag is None
    assert result.lower_bound == fluid_bound_heuristic(WEIGHTS)(root, inst)
