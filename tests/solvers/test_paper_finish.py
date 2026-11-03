"""MRCM finishing from a mid-episode state (`MrcmFinish`, D-55): on an empty chip it is MRCM's
own plan; from any state it gives a legal completion that reuses what is on chip; a problem MRCM
does not take gives None, with the reason counted."""

from __future__ import annotations

import pytest

from meda.core.composition import Composition, Rational
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance
from meda.core.rules import apply, replay
from meda.core.state import PrepState
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines.mrcm import mrcm
from meda.solvers.baselines.paper_finish import MrcmFinish
from meda.solvers.baselines.paper_graph import PaperSolver
from tests.core.helpers import load_yaml
from tests.solvers.test_emrcm import instance as emrcm_instance

GEOMETRIC = CostWeights.from_dict(load_yaml("costs/geometric.yaml"))
FAST = ("val_s2-011", "val_s2-014", "val_s2-020", "val_s2-024", "val_s2-036")  # MRCM < 1 s
S2 = [i for i in BenchmarkSet.read("val_s2_v2").instances if i.instance_id in FAST]


def mrcm_cost(inst: Instance) -> Rational:
    res = PaperSolver("mrcm", lambda x: mrcm(x, GEOMETRIC), GEOMETRIC).solve(inst)
    assert res.status == "feasible" and res.cost is not None
    return res.cost


def valid(inst: Instance, actions: tuple[object, ...]) -> None:
    _, totals, deltas = replay(inst, actions)  # type: ignore[arg-type]
    report = validate(MixingDAG.from_deltas(inst, deltas, totals, "t"), inst)
    assert report.ok, report


@pytest.mark.parametrize("inst", S2, ids=lambda i: i.instance_id)
def test_on_an_empty_chip_it_is_mrcms_own_plan(inst: Instance) -> None:
    finish = MrcmFinish(inst, GEOMETRIC)
    found = finish(PrepState.initial(inst))
    assert found is not None
    assert found[1] == mrcm_cost(inst)
    valid(inst, found[0])
    again = finish(PrepState.initial(inst))  # from the cached models: the same plan
    assert again == found


@pytest.mark.parametrize("inst", S2, ids=lambda i: i.instance_id)
def test_from_any_prefix_of_its_plan_it_finishes_legally(inst: Instance) -> None:
    """Legal from every prefix. Not always as cheap as MRCM's own rest: a prefix that stops
    inside an n-input mix leaves a part-mixed droplet MRCM never names, so it is rebuilt."""
    finish = MrcmFinish(inst, GEOMETRIC)
    start = finish(PrepState.initial(inst))
    assert start is not None
    plan, total = start
    state = PrepState.initial(inst)
    for k, action in enumerate(plan[:-1], start=1):
        state, _, _ = apply(state, action, inst)
        found = finish(state)
        assert found is not None, k
        valid(inst, plan[:k] + found[0])
    _, so_far, _ = replay(inst, plan[:-1])
    assert len(found[0]) == 1  # only the output was left, taken from the target droplet
    assert GEOMETRIC.scalar(so_far, inst.fluids) + found[1] == total


def test_a_target_already_on_chip_is_output_directly() -> None:
    inst = S2[0]
    finish = MrcmFinish(inst, GEOMETRIC)
    plan, _ = finish(PrepState.initial(inst)) or ((), 0)
    state = PrepState.initial(inst)
    for action in plan[:-1]:
        state, _, _ = apply(state, action, inst)
    found = finish(state)
    assert found is not None and found[0] == plan[-1:]


def test_a_problem_mrcm_does_not_take_gives_none_with_the_reason() -> None:
    inst = emrcm_instance([(Composition.from_ints((1, 3, 4), 8), 2, 2)])  # demand 2
    finish = MrcmFinish(inst, GEOMETRIC)
    assert finish.applicable() is not None
    assert finish(PrepState.initial(inst)) is None
    assert sum(finish.fails.values()) == 1
