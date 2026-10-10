"""The naive baseline: construction scope, exactness, and the SPEC 17.3 numbers.

`naive.py` is listed under M5 in the SPEC milestones, but M4's acceptance criterion ("every
sampled instance is solvable, checked with the naive solver") needs it, so it is built and
tested here. M5 adds the exact search that these plans are compared against.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meda.core.composition import Composition, Rational
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.solvers.base import PlanBuilder
from meda.solvers.baselines.naive import (
    NaiveSolver,
    factorise_within,
    mixed_radix_digits,
    naive_plan,
    plan_length_upper_bound,
)
from tests.core.helpers import chip, default_weights, dilution, mixture

WEIGHTS = default_weights()


def solve(instance: Instance) -> tuple[int, int, tuple[int, ...], int, Rational]:
    """(actions, mixes, reagent by fluid, waste, cost) of the naive plan, DAG-validated."""
    plan = naive_plan(instance)
    assert plan is not None, f"no naive plan for {instance.instance_id}"
    state, totals, deltas = replay(instance, plan)
    assert state.is_success
    dag = MixingDAG.from_deltas(instance, deltas, totals, "naive")
    validate(dag, instance).raise_if_invalid()
    return (
        len(plan),
        totals.mix_ops,
        totals.reagent_by_fluid,
        totals.waste_volume,
        WEIGHTS.scalar(totals, instance.fluids),
    )


# --------------------------------------------------------------------------- SPEC 17.3


def test_case1_quarter_m2() -> None:
    _, mixes, reagent, waste, cost = solve(dilution([(1, 4)], chip_limits=chip("dmfb_m2")))
    assert (mixes, reagent, waste) == (2, (1, 2), 2)
    assert cost == Rational.of(15)


def test_case2_quarter_m4() -> None:
    _, mixes, reagent, waste, cost = solve(dilution([(1, 4)], chip_limits=chip("meda_m4")))
    assert (mixes, reagent, waste) == (1, (1, 3), 3)
    assert cost == Rational.of(31, 2)


def test_case3_three_tenths_m8_and_infeasible_on_dmfb() -> None:
    _, mixes, reagent, waste, _ = solve(dilution([(3, 10)]))
    assert mixes == 3  # the optimum is 2 (sharing a partial take); naive does not share
    assert reagent == (2, 4)
    assert waste == 5
    # 3/10 is not dyadic: unreachable under dmfb_m2 with tolerance 0.
    assert naive_plan(dilution([(3, 10)], chip_limits=chip("dmfb_m2"))) is None


def test_case4_sharing_m2_naive_uses_four_mixes() -> None:
    _, mixes, reagent, waste, _ = solve(dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2")))
    assert mixes == 4  # SPEC 17.3 case 4: the optimum shares the 1/2 droplet and uses 3
    assert reagent == (3, 3)
    assert waste == 4


# --------------------------------------------------------------------------- scope


def test_pure_target_comes_straight_from_the_reservoir() -> None:
    actions, mixes, reagent, waste, cost = solve(mixture([(1, 0, 0)]))
    assert (actions, mixes, reagent, waste) == (1, 0, (1, 0, 0), 0)
    assert cost == Rational.of(10)


def test_chain_builds_a_mixture_in_one_pass() -> None:
    _, mixes, reagent, _, _ = solve(mixture([(2, 1, 1)]))  # den 4 <= M: 2 mixes for 3 fluids
    assert mixes == 2
    assert reagent == (2, 1, 1)


def test_demand_is_met_droplet_by_droplet() -> None:
    one = solve(dilution([(1, 4)], chip_limits=chip("dmfb_m2")))
    three = solve(dilution([(1, 4)], chip_limits=chip("dmfb_m2"), demand=3, step_limit=60))
    assert three[1] == 3 * one[1]  # no sharing: cost scales with demand
    assert three[2] == tuple(3 * r for r in one[2])


def test_out_of_scope_returns_none_not_a_claim() -> None:
    # A mixture whose denominator exceeds M: the chain cannot build it and the mixed-radix
    # construction is two-fluid only. Not found is not a claim of infeasibility.
    big = mixture([(1, 1, 1)])  # den 3 <= 8, buildable
    assert naive_plan(big) is not None
    hard = mixture([(5, 3, 2), (1, 1, 1)])  # den 10 > M for a 3-fluid chain
    assert naive_plan(hard) is None
    assert NaiveSolver(WEIGHTS).solve(hard).status == "infeasible"
    # A prime factor above M can never be a mix volume.
    assert naive_plan(dilution([(1, 11)])) is None


def test_step_limit_too_small_returns_none() -> None:
    assert naive_plan(dilution([(1, 32)], chip_limits=chip("dmfb_m2"), step_limit=3)) is None


def test_partial_take_is_required_for_the_radix_construction() -> None:
    """Carrying 1 unit of the previous droplet needs a partial take (SPEC 5.6)."""
    assert naive_plan(dilution([(1, 8)], chip_limits=chip("dmfb_m2"))) is not None
    strict = chip("dmfb_m2", allow_partial_take=False)
    assert naive_plan(dilution([(1, 8)], chip_limits=strict)) is None


def test_solver_result_fields() -> None:
    inst = dilution([(1, 4)], chip_limits=chip("dmfb_m2"))
    result = NaiveSolver(WEIGHTS).solve(inst)
    assert result.producer == "naive"
    assert result.status == "feasible"
    assert result.instance_id == inst.instance_id
    assert result.dag is not None and result.dag.producer == "naive"
    assert result.cost == Rational.of(15)
    assert result.totals is not None and result.totals.outputs == 1
    assert result.wall_time >= 0


# --------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("n", "lo", "hi", "expected"),
    [
        (4, 2, 2, (2, 2)),
        (8, 2, 2, (2, 2, 2)),
        (10, 2, 8, (2, 5)),
        (64, 2, 8, (8, 8)),
        (256, 2, 8, (4, 8, 8)),
        (7, 2, 8, (7,)),
        (11, 2, 8, None),  # prime above the mix limit
        (10, 2, 2, None),  # 5 cannot be reached with volume-2 mixes
        (1, 2, 8, None),
    ],
)
def test_factorise_within(n: int, lo: int, hi: int, expected: tuple[int, ...] | None) -> None:
    got = factorise_within(n, lo, hi)
    if expected is None:
        assert got is None
    else:
        assert got is not None
        assert sorted(got) == sorted(expected)
        product = 1
        for f in got:
            product *= f
        assert product == n
        assert all(lo <= f <= hi for f in got)


def test_mixed_radix_digits() -> None:
    assert mixed_radix_digits(1, (2, 2)) == (1, 0)
    assert mixed_radix_digits(3, (2, 2)) == (1, 1)
    assert mixed_radix_digits(3, (2, 5)) == (1, 1)
    assert mixed_radix_digits(13, (2, 2, 2, 2, 2)) == (1, 0, 1, 1, 0)
    with pytest.raises(ValueError, match="does not fit"):
        mixed_radix_digits(9, (2, 2))


def test_plan_length_bound_is_an_upper_bound() -> None:
    for inst in (
        dilution([(1, 4)], chip_limits=chip("dmfb_m2")),
        dilution([(3, 32)], chip_limits=chip("dmfb_m2"), step_limit=80),
        dilution([(3, 10)], demand=2, step_limit=60),
        mixture([(2, 1, 1)], demand=2, step_limit=60),
    ):
        plan = naive_plan(inst)
        assert plan is not None
        assert len(plan) <= plan_length_upper_bound(inst)


def test_plan_builder_rejects_illegal_steps() -> None:
    from meda.core.actions import Source
    from meda.core.rules import IllegalActionError

    builder = PlanBuilder.start(dilution([(1, 4)], chip_limits=chip("dmfb_m2")))
    with pytest.raises(IllegalActionError):
        builder.mix(Source.reservoir(0), 5, Source.reservoir(1), 4)
    assert builder.actions == []


# --------------------------------------------------------------------------- property


@settings(max_examples=40, deadline=None)
@given(numerator=st.integers(1, 31))
def test_every_dilution_over_32_is_built_exactly_at_m2(numerator: int) -> None:
    """The mixed-radix construction is exact for every n/32 under 1:1 mixing."""
    inst = dilution([(numerator, 32)], chip_limits=chip("dmfb_m2"), step_limit=200)
    plan = naive_plan(inst)
    assert plan is not None
    state, totals, deltas = replay(inst, plan)
    assert state.is_success
    validate(MixingDAG.from_deltas(inst, deltas, totals, "naive"), inst).raise_if_invalid()


@settings(max_examples=30, deadline=None)
@given(parts=st.lists(st.integers(1, 3), min_size=3, max_size=3))
def test_small_mixtures_are_built_exactly(parts: list[int]) -> None:
    inst = mixture([tuple(parts)], step_limit=60)
    plan = naive_plan(inst)
    target = Composition.from_ratio(tuple(parts))
    if target.den > inst.chip.max_mix_volume:
        return  # out of the chain's scope
    assert plan is not None
    state, totals, deltas = replay(inst, plan)
    assert state.is_success
    outputs = [
        n
        for n in MixingDAG.from_deltas(inst, deltas, totals, "naive").nodes
        if n.target_id is not None
    ]
    assert all(n.composition == target for n in outputs)


def test_tolerance_does_not_change_the_plan() -> None:
    """The construction is exact, so a loose tolerance costs the same (it never approximates)."""
    exact = solve(dilution([(1, 8)], chip_limits=chip("dmfb_m2")))
    loose = solve(
        replace(
            dilution([(1, 8)], chip_limits=chip("dmfb_m2")),
            targets=(
                replace(
                    dilution([(1, 8)], chip_limits=chip("dmfb_m2")).targets[0],
                    tolerance=Rational.of(1, 4),
                ),
            ),
        )
    )
    assert exact == loose
