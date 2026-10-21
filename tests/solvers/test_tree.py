"""The split-tree baseline (SPEC 10.4): builds every reachable target, over any number of
fluids and any M, and is never cheaper than a proved optimum."""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from meda.core.actions import Mix
from meda.core.composition import ZERO, Composition, Rational, linf_distance
from meda.core.dag import MixingDAG, validate
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import apply, replay
from meda.core.state import PrepState
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.base import SolveBudget
from meda.solvers.baselines.naive import factorise_within, naive_plan
from meda.solvers.baselines.tree import (
    TreeSolver,
    aim,
    dilute_up_input,
    min_denominator_within,
    split_pieces,
    tree_plan,
    within_tolerance,
)
from meda.solvers.exact_search import ExactSearch
from tests.core.helpers import chip, default_weights, dilution, mixture

WEIGHTS = default_weights()


def fluids(count: int) -> tuple[Fluid, ...]:
    kinds = [FluidKind.SAMPLE, *[FluidKind.REAGENT] * (count - 2), FluidKind.BUFFER]
    return tuple(Fluid(i, f"F{i}", k) for i, k in enumerate(kinds))


def single(num: tuple[int, ...], den: int, m: int, **target: int) -> Instance:
    t = Target(
        0,
        0,
        Composition.from_ints(num, den),
        ZERO,
        target.get("volume", 1),
        target.get("demand", 1),
    )
    return Instance(
        "test-tree", fluids(len(num)), (t,), chip(max_mix_volume=m, **_chip(target)), 10_000
    )


def _chip(target: dict[str, int]) -> dict[str, int]:
    return {"storage_capacity": target["storage"]} if "storage" in target else {}


def checked(instance: Instance) -> MixingDAG:
    """The tree plan replays to success and its DAG validates."""
    plan = tree_plan(instance)
    assert plan is not None, instance.targets
    final, totals, deltas = replay(instance, plan)
    assert final.is_success
    dag = MixingDAG.from_deltas(instance, deltas, totals, "tree")
    validate(dag, instance).raise_if_invalid()
    return dag


def test_the_worked_example_is_two_mixes() -> None:
    """3/16 S + 5/16 R + 1/2 B at M = 8: Mix(S 3, R 5), then 1 unit of that + 1 of B."""
    inst = single((3, 5, 8), 16, 8)
    assert naive_plan(inst) is None
    dag = checked(inst)
    assert sum(1 for n in dag.nodes if n.type.name == "MIX") == 2


@pytest.mark.parametrize(("e", "num", "den"), [(2, (3, 5, 8), 16), (4, (1, 2, 3, 2), 8)])
def test_split_pieces_blend_back_to_the_target(e: int, num: tuple[int, ...], den: int) -> None:
    comp = Composition.from_ints(num, den)
    pieces = split_pieces(comp, e)
    assert sum(count for _, count in pieces) == e
    blend = [
        sum(count * p.num[f] * (den // e // p.den) for p, count in pieces) for f in range(len(num))
    ]
    assert Composition.from_ints(tuple(blend), den) == comp


@pytest.mark.parametrize(
    ("num_fluids", "den", "m"),
    [(2, 48, 3), (3, 16, 4), (3, 30, 5), (3, 32, 8), (4, 16, 8), (4, 8, 2)],
)
def test_builds_exactly_the_reachable_targets(num_fluids: int, den: int, m: int) -> None:
    """Every target whose denominator has no prime factor above M is built; no other is."""
    for parts in itertools.product(range(den + 1), repeat=num_fluids - 1):
        if sum(parts) >= den:
            continue
        num = (*parts, den - sum(parts))
        comp = Composition.from_ints(num, den)
        if comp.den == 1:
            continue
        inst = single(num, den, m)
        if factorise_within(comp.den, 2, m) is None:
            assert tree_plan(inst) is None
        else:
            checked(inst)


def test_an_unreachable_target_is_not_found() -> None:
    assert tree_plan(single((5, 4, 2), 11, 8)) is None  # prime 11 > M
    assert tree_plan(single((10, 39), 49, 5)) is None  # 7 > M = 5
    checked(single((10, 39), 49, 7))


def test_demand_shares_a_droplet_and_volume_scales_the_top_mix() -> None:
    three = checked(single((3, 5, 8), 16, 8, demand=3))
    outputs = [n for n in three.nodes if n.type.name == "OUTPUT"]
    assert len(outputs) == 3
    assert sum(1 for n in three.nodes if n.type.name == "MIX") == 2  # one droplet serves all
    big = checked(single((3, 5, 8), 16, 8, volume=3))
    assert all(n.volume == 3 for n in big.nodes if n.type.name == "OUTPUT")


def test_a_large_droplet_is_made_by_a_dilute_up() -> None:
    """1/9 at volume 8 on M = 8: no split fits (a top mix must be a multiple of 3, so 3 or 6
    units), so the last mix is 1 unit of 8/9 with 7 of buffer."""
    inst = single((1, 8), 9, 8, volume=8)
    assert naive_plan(inst) is None
    dag = checked(inst)
    last = [n for n in dag.nodes if n.type.name == "MIX"][-1]
    assert last.volume == 8
    assert all(n.volume == 8 for n in dag.nodes if n.type.name == "OUTPUT")


@pytest.mark.parametrize(
    ("num", "den", "s", "fluid", "l"), [((1, 8), 9, 8, 1, 7), ((3, 5, 8), 16, 7, 2, 3)]
)
def test_dilute_up_input_mixes_back_to_the_target(
    num: tuple[int, ...], den: int, s: int, fluid: int, l: int
) -> None:
    comp = Composition.from_ints(num, den)
    xa = dilute_up_input(comp, s, fluid, l)
    assert xa is not None
    k = s - l
    blend = [k * xa.num[f] * den // xa.den for f in range(len(num))]
    blend[fluid] += l * den
    assert Composition.from_ints(tuple(blend), s * den) == comp
    assert dilute_up_input(comp, 8, 0, 7) is None  # 7 units of S exceed 8 x 1/9


@pytest.mark.parametrize("m", [3, 5, 8, 10])
def test_every_reachable_dilution_is_built_at_every_volume_up_to_m(m: int) -> None:
    """The split alone stops near M / 2: its top mix must be a multiple of a factor of D."""
    for den in (6, 8, 9, 12, 16, 20, 24, 30, 32, 36):
        if factorise_within(den, 2, m) is None:
            continue
        for n in range(1, den):
            if Composition.from_ints((n, den - n), den).den != den:
                continue
            for volume in range(1, m + 1):
                assert tree_plan(single((n, den - n), den, m, volume=volume)) is not None, (
                    n,
                    den,
                    volume,
                )


def test_a_mixture_at_volume_m_is_made_by_a_blend_of_two_built_droplets() -> None:
    """5/16, 5/16, 3/16, 3/16 at volume 3 on M = 3: a dilute-up's built input would still hold
    all four fluids and need a volume-2 split it cannot fit, so the last mix blends two built
    droplets (2 units of one, 1 of the other)."""
    inst = single((5, 5, 3, 3), 16, 3, volume=3)
    plan = tree_plan(inst)
    assert plan is not None
    last = [a for a in plan if isinstance(a, Mix)][-1]
    assert not last.a.is_reservoir and not last.b.is_reservoir and last.k + last.l == 3
    checked(inst)
    # Found before by a dilute-up only with intermediates finer than 1/256; a blend needs none.
    checked(single((13, 25, 26), 64, 7, volume=7))


def test_tight_storage_picks_a_recipe_that_fits() -> None:
    """The recipe's storage peak is computed exactly and must fit the chip, so at capacity 2
    every 3-fluid 64th of this sample is built on M = 8 (the split alone needs 3 for some),
    and the plan really never holds more than 2 droplets."""
    for a in range(1, 30, 3):
        for b in range(1, 30, 5):
            inst = single((a, b, 64 - a - b), 64, 8, storage=2)
            plan = tree_plan(inst)
            assert plan is not None, (a, b)
            state = PrepState.initial(inst)
            for action in plan:
                state, _, _ = apply(state, action, inst)
                assert len(state.droplets) <= 2


def test_a_target_with_no_recipe_in_budget_is_not_found_rather_than_raising() -> None:
    """Six fluids in 16ths on M = 3 at capacity 2: no recipe within the search budget."""
    assert tree_plan(single((3, 3, 3, 3, 2, 2), 16, 3, storage=2)) is None


@pytest.mark.parametrize(
    "instance", BenchmarkSet.read("tiny_exact_v1").instances, ids=lambda i: i.instance_id
)
def test_never_cheaper_than_a_proved_optimum(instance: Instance) -> None:
    result = TreeSolver(WEIGHTS).solve(instance)
    ref = BenchmarkSet.read("tiny_exact_v1").reference[instance.instance_id]
    if result.cost is not None:
        assert result.cost >= ref.cost


@pytest.mark.parametrize(
    "ratio",
    [
        (1, 1, 2),
        pytest.param((3, 1, 4), marks=pytest.mark.slow),  # about a minute of exact search
        pytest.param((1, 2, 1, 4), marks=pytest.mark.slow),
    ],
)
def test_never_below_exact_search_on_small_mixtures(ratio: tuple[int, ...]) -> None:
    inst = (
        mixture([ratio])
        if len(ratio) == 3
        else Instance(
            "test-four",
            fluids(4),
            (Target(0, 0, Composition.from_ratio(ratio), ZERO, 1, 1),),
            chip(),
            30,
        )
    )
    den = Composition.from_ratio(ratio).den
    inst = replace(
        inst, chip=replace(inst.chip, max_mix_volume=4, max_denominator=den), step_limit=12
    )
    tree = TreeSolver(WEIGHTS).solve(inst)
    exact = ExactSearch(WEIGHTS).solve(inst, SolveBudget(max_nodes=20_000))
    assert tree.cost is not None
    # A proved optimum, or on a budget stop the proved lower bound; the tree is a real plan,
    # so it can never be below either.
    floor = exact.cost if exact.status == "optimal" else exact.lower_bound
    assert floor is not None and tree.cost >= floor
    if exact.status == "timeout":
        assert exact.cost is not None and exact.cost <= tree.cost  # seeded with the tree


def test_on_dilutions_it_agrees_with_reachability_like_naive() -> None:
    for n in range(1, 16):
        inst = dilution([(n, 16)], chip_limits=chip("dmfb_m2"), step_limit=200)
        assert (tree_plan(inst) is None) == (naive_plan(inst) is None)


@pytest.mark.parametrize(
    ("num", "den", "tol", "grid"),
    [((70, 49), 119, (1, 200), 12), ((3, 7), 10, (1, 20), 6), ((1, 2, 4), 7, (1, 50), 16)],
)
def test_within_tolerance_matches_brute_force(
    num: tuple[int, ...], den: int, tol: tuple[int, int], grid: int
) -> None:
    comp, eps = Composition.from_ints(num, den), Rational.of(*tol)
    for d in range(1, grid + 1):
        exists = any(
            linf_distance(Composition.from_ints((*parts, d - sum(parts)), d), comp) <= eps
            for parts in itertools.product(range(d + 1), repeat=len(num) - 1)
            if sum(parts) <= d
        )
        found = within_tolerance(comp, eps, d)
        assert (found is not None) == exists, d
        if found is not None:
            assert linf_distance(found, comp) <= eps and d % found.den == 0


def test_a_tolerant_target_is_built_at_the_coarsest_reachable_grid() -> None:
    """FacDA's example: 70:49 (= 70/119, 119 = 7 x 17) at M = 6 has no exact plan; within
    0.005 the coarsest reachable composition is 7/12 (FacDA's own rewrite is 71/120)."""
    target = Target(0, 0, Composition.from_ints((70, 49), 119), Rational.of(1, 200), 1, 1)
    inst = Instance("facda", fluids(2), (target,), chip(max_mix_volume=6), 1_000)
    assert aim(target, inst.chip) == Composition.from_ints((7, 5), 12)
    checked(inst)
    assert min_denominator_within(target.composition, target.tolerance, 256) == 12
    exact = replace(target, tolerance=ZERO)
    assert tree_plan(replace(inst, targets=(exact,))) is None  # 17 > M
