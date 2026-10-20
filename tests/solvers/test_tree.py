"""The split-tree baseline (SPEC 10.4): builds every reachable target, over any number of
fluids and any M, and is never cheaper than a proved optimum."""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from meda.core.composition import ZERO, Composition
from meda.core.dag import MixingDAG, validate
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.base import SolveBudget
from meda.solvers.baselines.naive import factorise_within, naive_plan
from meda.solvers.baselines.tree import TreeSolver, split_pieces, tree_plan
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


def test_tight_storage_returns_none_rather_than_raising() -> None:
    results = [
        tree_plan(single((a, b, 64 - a - b), 64, 8, storage=2))
        for a in range(1, 30, 3)
        for b in range(1, 30, 5)
    ]
    assert any(r is None for r in results) and any(r is not None for r in results)


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
