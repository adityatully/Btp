"""Fukui's droplet reuse on DFMM (ISOCC 2024) against the paper's Fig. 3, and legal plans that
reuse (SPEC 10.4). The reading is in `meda.solvers.baselines.fukui`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.dfmm import best_graph
from meda.solvers.baselines.fukui import fukui, partner
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from tests.core.helpers import SRB, chip


def instance(ratios: list[tuple[int, ...]], m: int, demand: int = 1) -> Instance:
    ts = tuple(
        Target(i, 0, Composition.from_ratio(r), ZERO, 1, demand) for i, r in enumerate(ratios)
    )
    return Instance(
        "fukui", SRB, ts, chip(max_mix_volume=m, storage_capacity=64, max_denominator=1024), 500
    )


def reagents(inst: Instance) -> int:
    graph = fukui(inst)
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "fukui"), inst)
    drawn = graph.counts(sum(t.demand for t in inst.targets))["drawn"]
    assert isinstance(drawn, tuple)
    return sum(drawn)


def dfmm_alone(ratios: list[tuple[int, ...]], m: int) -> int:
    total = 0
    for r in ratios:
        g = best_graph(r, 2, m)
        drawn = g.counts(g.volume(len(g.nodes) - 1))["drawn"]
        assert isinstance(drawn, tuple)
        total += sum(drawn)
    return total


def test_fig_3_partner() -> None:
    """P 4:3:5 from D_w <3:0:1> at 1 : 2 needs D_n 1:3:4."""
    got = partner(Composition.from_ratio((4, 3, 5)), Composition.from_ratio((3, 0, 1)), 1, 3)
    assert got == Composition.from_ratio((1, 3, 4))


def test_partner_negative_share() -> None:
    """Too much D_w would need a negative share of fluid 0."""
    assert (
        partner(Composition.from_ratio((1, 3, 8)), Composition.from_ratio((3, 0, 1)), 1, 2) is None
    )


@pytest.mark.parametrize("second", [(1, 3, 8), (1, 2, 9), (2, 5, 5)])
def test_reuse_saves_reagents(second: tuple[int, ...]) -> None:
    """7:7:6 (Fig. 1) then a 12-sum target at M 5: a leftover of the first graph is reused."""
    ratios = [(7, 7, 6), second]
    assert reagents(instance(ratios, 5)) < dfmm_alone(ratios, 5)


def test_one_target_is_dfmm() -> None:
    assert reagents(instance([(7, 7, 6)], 5)) == dfmm_alone([(7, 7, 6)], 5)


def test_demand_replays() -> None:
    reagents(instance([(7, 7, 6), (4, 3, 5)], 5, demand=3))


def test_not_applicable() -> None:
    """A ratio-sum of 7 has no factor within M 5, and no reuse is possible for the first target."""
    with pytest.raises(NotApplicable):
        fukui(instance([(3, 2, 2)], 5))
