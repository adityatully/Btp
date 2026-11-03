"""e-MRCM (Liang et al., TCAD 2020) reproduces its paper's worked example and gives legal plans
(plan 6.3, SPEC 10.4). Its reading of the paper is in `meda.solvers.baselines.emrcm`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition, Rational
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.emrcm import emrcm, emrcm_graph
from meda.solvers.baselines.mrcm import _Search
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from tests.core.helpers import chip, load_yaml
from tests.solvers.test_literature import SRB

GEOMETRIC = CostWeights.from_dict(load_yaml("costs/geometric.yaml"))
W = (Rational.of(2), Rational.of(1), Rational.of(3))  # Fig. 11: R3 dearest
R1 = Composition.from_ints((25, 31, 44), 100)
R2 = Composition.from_ints((75, 6, 44), 125)  # printed <3, 6, 44>; see the module docstring


def cost(drawn: object) -> Rational:
    assert isinstance(drawn, tuple)
    return sum((w * x for w, x in zip(W, drawn, strict=True)), Rational.of(0))


def instance(targets: list[tuple[Composition, int, int]]) -> Instance:
    ts = tuple(Target(i, 0, c, ZERO, v, d) for i, (c, v, d) in enumerate(targets))
    return Instance("emrcm", SRB, ts, chip(max_mix_volume=8, storage_capacity=32), 200)


def test_fig_11a_second_target_tree() -> None:
    """MRCM's own tree for the second target is Fig. 11's: 3 R1 with 2 of <0, 3, 22>, made of
    2 R3 with 3 of <0, 1, 4>; 3 mixes, cost 25, waste 6."""
    tree = _Search(W, 2, 8).best(R2, 4)
    assert tree is not None and tree.key == (Rational.of(25), 3, 6)
    assert tree.children[0][1].children[0][1].pure == (0, 1, 4)


def test_fig_11c_shared_graph() -> None:
    """Sharing <0, 3, 22>: 4 mixes, R1 4, R2 2, R3 6, waste 4, reactant cost 28."""
    graph = emrcm_graph([(R1, 4), (R2, 4)], W, 2, 8, 3)
    counts = graph.counts(8)
    assert counts == {"nm": 4, "drawn": (4, 2, 6), "nw": 4}
    assert cost(counts["drawn"]) == Rational.of(28)


def test_never_dearer_than_separate_trees() -> None:
    """Each target after the first may only gain from a shared droplet."""
    targets = [(R1, 4), (R2, 4), (Composition.from_ints((1, 3, 4), 8), 2)]
    shared = cost(emrcm_graph(targets, W, 2, 8, 3).counts(10)["drawn"])
    alone = Rational.of(0)
    for c, v in targets:
        tree = _Search(W, 2, 8).best(c, v)
        assert tree is not None
        alone = alone + tree.key[0]
    assert shared <= alone


def test_plan_replays_and_validates() -> None:
    inst = instance([(R1, 4, 1), (R2, 4, 1)])
    graph = emrcm(inst, GEOMETRIC)
    final, totals, deltas = replay(inst, emit(inst, graph))
    assert final.is_success
    validate(MixingDAG.from_deltas(inst, deltas, totals, "emrcm"), inst).raise_if_invalid()
    assert totals.reagent_by_fluid == graph.counts(8)["drawn"]


def test_equal_targets_share_one_droplet() -> None:
    graph = emrcm_graph([(R1, 1), (R1, 1)], W, 2, 8, 3)  # every mix makes at least 2
    assert graph.roots is not None and graph.roots[0] == graph.roots[1]


def test_demand_above_one_is_not_applicable() -> None:
    with pytest.raises(NotApplicable, match="demand 1"):
        emrcm(instance([(R1, 4, 2)]), GEOMETRIC)


def test_narrow_search_keeps_the_papers_trees() -> None:
    """Eq. 11 as we read it (a mix of T units needs T | D, children's denominators dividing D/T
    together) still finds every tree the paper draws: Fig. 3(a) and Fig. 11(c)."""
    fig_3a = _Search((Rational.of(64), Rational.of(8), Rational.of(1)), 2, 8, narrow=True)
    tree = fig_3a.best(R1, 4)
    assert tree is not None and tree.key[1:] == (3, 5)
    assert emrcm_graph([(R1, 4), (R2, 4)], W, 2, 8, 3, narrow=True).counts(8) == {
        "nm": 4,
        "drawn": (4, 2, 6),
        "nw": 4,
    }


def test_narrow_search_needs_mixes_that_divide_the_denominator() -> None:
    """A third needs a mix of 3 (or 6) units under Eq. 11; the full search may also halve."""
    third = Composition.from_ints((1, 2), 3)
    tree = _Search((Rational.of(64), Rational.of(1)), 2, 8, narrow=True).best(third, 2)
    assert tree is not None and tree.volume % 3 == 0
