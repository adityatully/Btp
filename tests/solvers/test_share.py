"""The sharing pass (`baselines.share`, SPEC 10.4): legal plans that reuse droplets across
targets, never dearer than the constructions, and at the proved optimum on the worked
examples of Track 2 E2 (paper log E-5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from meda.core.composition import ZERO, Rational
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance
from meda.core.rules import apply, replay
from meda.core.state import PrepState
from meda.generators.benchmark_io import BenchmarkSet
from meda.generators.sampler import GeneratorSource
from meda.generators.stages import StageDesign
from meda.solvers.baselines.share import (
    ShareSolver,
    SharingCostToFinish,
    finish_from,
    share_plan,
)
from meda.solvers.fallback import (
    RELAXED_STEP_LIMIT,
    ConstructiveCostToFinish,
    cheapest_construction,
)
from tests.core.helpers import chip, dilution, load_yaml

GEOMETRIC = CostWeights.from_dict(load_yaml("costs/geometric.yaml"))


def cost_of(instance: Instance, weights: CostWeights = GEOMETRIC) -> Rational:
    """The share plan replays to success, its DAG validates, and this is its cost."""
    plan = share_plan(instance, weights)
    assert plan is not None
    final, totals, deltas = replay(instance, plan)
    assert final.is_success
    report = validate(MixingDAG.from_deltas(instance, deltas, totals, "share"), instance)
    assert report.ok, report
    return weights.scalar(totals, instance.fluids)


def construction_cost(instance: Instance, weights: CostWeights = GEOMETRIC) -> Rational:
    best = cheapest_construction(instance, weights)
    assert best is not None
    return best[1]


def test_two_targets_share_one_half_droplet() -> None:
    """E-5 worked example 1: M = 3, 3 units of 1/6 and 2 units of 3/4. The tree pays for
    3 sample units (202); the optimum (134, proved) shares a 1/2 droplet and draws 2."""
    inst = dilution([(1, 6), (3, 4)], chip(max_mix_volume=3), step_limit=24)
    inst = replace(
        inst,
        targets=(
            replace(inst.targets[0], droplet_volume=3),
            replace(inst.targets[1], droplet_volume=2),
        ),
    )
    assert construction_cost(inst) == Rational.of(202)
    assert cost_of(inst) == Rational.of(134)


def test_a_cheaper_chain_under_tight_storage() -> None:
    """3/4 at volume 2 on M = 6 with storage 2: the tree mixes 3 S + 1 B (196); the proved
    optimum goes through 1/2 and draws 2 sample units (132)."""
    inst = dilution([(3, 4)], chip(max_mix_volume=6, storage_capacity=2), droplet_volume=2)
    assert construction_cost(inst) == Rational.of(196)
    assert cost_of(inst) == Rational.of(132)


def test_a_gradient_with_demand_reuses_leftovers() -> None:
    """E-5 worked example 2's cell (gradient, demand, storage 2): 3 x 2 units of 1/8 and
    2 x 3 units of 3/8 at M = 3. Tree 496, proved optimum 211; the pass gets 280."""
    inst = dilution([(1, 8), (3, 8)], chip(max_mix_volume=3, storage_capacity=2), step_limit=60)
    inst = replace(
        inst,
        targets=(
            replace(inst.targets[0], droplet_volume=2, demand=3),
            replace(inst.targets[1], droplet_volume=3, demand=2),
        ),
    )
    assert construction_cost(inst) == Rational.of(496)
    assert Rational.of(211) <= cost_of(inst) <= Rational.of(280)


def test_never_dearer_than_the_constructions_and_never_below_an_optimum() -> None:
    bench = BenchmarkSet.read("tiny_exact_v1")
    weights = CostWeights.from_dict(load_yaml("costs/default.yaml"))
    for inst in bench.instances:
        ref = bench.reference[inst.instance_id]
        assert ref.status == "optimal"
        cost = cost_of(inst, weights)
        assert ref.cost <= cost <= construction_cost(inst, weights), inst.instance_id


def _stage_draws(stage: str, count: int, seed: int) -> list[Instance]:
    design = StageDesign.from_dict(load_yaml("stages/track2_v1.yaml"))
    cells = next(s for s in design.stages if s.name == stage).cells
    rng = np.random.default_rng(seed)
    return [
        GeneratorSource(cells[i % len(cells)].config, chip(), GEOMETRIC)(rng) for i in range(count)
    ]


@pytest.mark.parametrize("stage", ["s3_demand", "s4_targets", "s5_storage"])
def test_legal_and_never_dearer_on_stage_draws(stage: str) -> None:
    """Fresh problems of the sharing stages: every plan is legal (storage and step limit
    included, since it is replayed through the engine) and never dearer than the teacher."""
    for inst in _stage_draws(stage, 4 if stage == "s4_targets" else 6, seed=7):
        assert cost_of(inst) <= construction_cost(inst), inst.instance_id


def test_held_out_cell_draws() -> None:
    design = StageDesign.from_dict(load_yaml("stages/track2_v1.yaml"))
    rng = np.random.default_rng(11)
    for cell in design.held_out:
        for _ in range(3):
            inst = GeneratorSource(cell.config, chip(), GEOMETRIC)(rng)
            assert cost_of(inst) <= construction_cost(inst)


def test_lookahead_plans_are_legal() -> None:
    for inst in _stage_draws("s5_storage", 3, seed=3):
        plan = share_plan(inst, GEOMETRIC, lookahead=2)
        assert plan is not None
        final, _, _ = replay(inst, plan)
        assert final.is_success


def test_solver_reports_a_feasible_plan() -> None:
    inst = dilution([(1, 6), (3, 4)], chip(max_mix_volume=3), step_limit=24)
    result = ShareSolver(GEOMETRIC).solve(inst)
    assert result.status == "feasible"
    assert result.dag is not None and result.cost is not None


@pytest.mark.parametrize("name", ["val_s1_v2", "val_s3_v2"])
def test_finish_from_any_state_reuses_the_chip_and_never_costs_more_than_rebuilding(
    name: str,
) -> None:
    """From every state along the sharing pass's own plan, `finish_from` returns a plan that
    replays from that state to success at the returned cost, and the penalty built on it is
    never dearer than wasting the chip and rebuilding (paper log E-18)."""
    rebuild = ConstructiveCostToFinish(GEOMETRIC)
    penalty = SharingCostToFinish(GEOMETRIC)
    for inst in BenchmarkSet.read(name).instances[:8]:
        plan = share_plan(inst, GEOMETRIC)
        assert plan is not None
        relaxed = replace(inst, step_limit=RELAXED_STEP_LIMIT)
        state = PrepState.initial(relaxed)
        for action in plan[:-1]:
            state, _, _ = apply(state, action, relaxed)
            found = finish_from(state, inst, GEOMETRIC)
            assert found is not None
            rest, cost = found
            end, total = state, ZERO
            for a in rest:
                end, step, _ = apply(end, a, relaxed)
                total = total + GEOMETRIC.scalar(step, inst.fluids)
            assert end.is_success and total == cost
            assert penalty(state, inst) <= rebuild(state, inst)


def test_penalty_at_the_start_is_never_dearer_than_the_construction() -> None:
    """`finish_from` alone is greedy and can be dearer (val_s1-003: 465 vs 402); the penalty
    takes the cheaper of it and rebuilding, so it never is."""
    penalty = SharingCostToFinish(GEOMETRIC)
    for inst in BenchmarkSet.read("val_s1_v2").instances[:10]:
        assert penalty(PrepState.initial(inst), inst) <= construction_cost(inst)
