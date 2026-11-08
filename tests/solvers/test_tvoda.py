"""TVODA (Kundu et al., TCAD 2022) against the paper's Fig. 9, and legal plans (SPEC 10.4).
The reading is in `meda.solvers.baselines.tvoda`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from meda.solvers.baselines.tvoda import forest, tvoda
from tests.core.helpers import SB, chip


def instance(x: int, den: int, demand: int, m: int, v: int = 1) -> Instance:
    t = Target(0, 0, Composition.from_ints((x, den - x), den), ZERO, v, demand)
    return Instance("tvoda", SB, (t,), chip(max_mix_volume=m, storage_capacity=64), 500)


def test_fig_9_forest() -> None:
    assert forest(120, 2, 6, 13, 1) == [[4, 5, 6], [4, 5, 6], [4, 5, 3, 2]]


def test_fig_9_no_worse_than_the_paper() -> None:
    """71 : 49, M 6, Vt 13: the paper's nm 9, ns 9, nw 6; ours 7, 9, 4."""
    inst = instance(71, 120, 13, 6)
    graph = tvoda(inst)
    counts = graph.counts(13)
    assert (counts["nm"], counts["drawn"], counts["nw"]) == (7, (9, 8), 4)
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "tvoda"), inst)


@pytest.mark.parametrize(("x", "den", "demand", "m", "v"), [(3, 8, 5, 4, 1), (7, 20, 3, 5, 2)])
def test_plans_replay_and_validate(x: int, den: int, demand: int, m: int, v: int) -> None:
    inst = instance(x, den, demand, m, v)
    _, totals, deltas = replay(inst, emit(inst, tvoda(inst)))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "tvoda"), inst)


def test_a_prime_above_m_is_not_applicable() -> None:
    with pytest.raises(NotApplicable):
        tvoda(instance(3, 7, 2, 6))
