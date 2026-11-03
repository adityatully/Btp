"""NFSP (Dinh, Yamashita and Ho, ASP-DAC 2014) reproduces its paper's worked example and gives
legal plans (plan 6.3, SPEC 10.4). Its reading of the paper is in `meda.solvers.baselines.nfsp`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition, Rational
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.nfsp import Want, flow, graph_of, nfsp, precision
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from tests.core.helpers import SB, chip, load_yaml

GEOMETRIC = CostWeights.from_dict(load_yaml("costs/geometric.yaml"))


def instance(targets: list[tuple[Composition, Rational, int, int]], m: int = 2) -> Instance:
    ts = tuple(Target(i, 0, c, tol, v, k) for i, (c, tol, v, k) in enumerate(targets))
    return Instance("nfsp", SB, ts, chip(max_mix_volume=m, storage_capacity=64), 200)


def eighths(x: int) -> Composition:
    return Composition.from_ints((x, 8 - x), 8)


def test_fig_2_flow() -> None:
    """d = 3, four droplets of 3/8 and six of 5/8, cost_s 2, cost_b 1: sample 6, buffer 5,
    F = 17, one waste droplet."""
    found = flow(3, [Want(3, 1, 4), Want(5, 1, 6)], 2, 1, 1)
    assert (found.sample, found.buffer) == (6, 5)
    assert 2 * found.sample + found.buffer == 17


def test_fig_2_plan_replays_and_validates() -> None:
    inst = instance([(eighths(3), ZERO, 1, 4), (eighths(5), ZERO, 1, 6)])
    graph = nfsp(inst, GEOMETRIC)
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "nfsp"), inst)
    assert totals.reagent_by_fluid == (6, 5)


def test_meda_mixes_hold_whole_target_droplets() -> None:
    """A droplet of 3 units comes from one mix of at least 2:2 units."""
    inst = instance([(eighths(3), ZERO, 3, 2)], m=4)
    graph = nfsp(inst, GEOMETRIC)
    assert all(graph.volume(r) >= 3 for r in graph.root_nodes())
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "nfsp"), inst)


def test_tolerant_targets_move_to_the_coarsest_grid() -> None:
    """1/3 within 1/20: 3/8 is 1/24 away, 5/16 is 1/48 away; d = 3 is the coarsest that fits."""
    third = Composition.from_ints((1, 2), 3)
    d, wants = precision(instance([(third, Rational.of(1, 20), 1, 1)]))
    assert (d, wants) == (3, [Want(3, 1, 1)])


def test_exact_target_off_the_grid_is_not_applicable() -> None:
    third = Composition.from_ints((1, 2), 3)
    with pytest.raises(NotApplicable, match="grid"):
        precision(instance([(third, ZERO, 1, 1)]))


def test_flow_is_drawn_in_level_order() -> None:
    """Every node draws only on earlier nodes and delivers what the flow promised."""
    wants = [Want(3, 1, 4), Want(5, 1, 6)]
    graph = graph_of(flow(3, wants, 2, 1, 1), wants, 0, 1, 1)
    assert len(graph.root_nodes()) == 10
    assert graph.counts(10)["drawn"] == (6, 5)
