"""STWM and MTWM (Kundu & Roy, TODAES 2023) against the paper's worked examples, and legal plans
(SPEC 10.4). The reading of the paper is in `meda.solvers.baselines.mtwm`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition
from meda.core.dag import MixingDAG, validate
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.mtwm import Node, allowed, mtwm, stwm
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, emit
from tests.core.helpers import chip


def fluids(t: int) -> tuple[Fluid, ...]:
    kind = [FluidKind.SAMPLE] + [FluidKind.REAGENT] * (t - 2) + [FluidKind.BUFFER]
    return tuple(Fluid(i, f"R{i + 1}", k) for i, k in enumerate(kind))


def instance(ratios: list[tuple[int, ...]], m: int, demand: int = 1) -> Instance:
    targets = tuple(
        Target(i, 0, Composition.from_ints(r, sum(r)), ZERO, 1, demand)
        for i, r in enumerate(ratios)
    )
    return Instance(
        "mtwm", fluids(len(ratios[0])), targets, chip(max_mix_volume=m, storage_capacity=64), 500
    )


def paper_counts(graph: PaperGraph) -> tuple[int, int, int]:
    """(nw, nr, nm) as the paper counts them: a root's whole volume is the target."""
    c = graph.counts(sum(graph.volume(r) for r in graph.root_nodes()))
    nw, drawn, nm = c["nw"], c["drawn"], c["nm"]
    assert isinstance(nw, int) and isinstance(drawn, tuple) and isinstance(nm, int)
    return nw, sum(drawn), nm


FIG_15 = [(2, 11, 5), (12, 5, 1), (5, 6, 14)]


def test_fig_15b_graph_is_one_the_model_allows() -> None:
    """The paper's MTWM graph uses only edges the sharing rule allows on our DFMM skeletons and
    makes the three targets, so our optimum can only be as good or better. It prints nw 8, nr 20,
    nm 8; its waste dots give 0:3:2 two, but that node makes 5 units and passes on 2, so nw is 9.
    """
    graph = PaperGraph(
        3,
        (
            (In(1, fluid=1), In(1, fluid=2)),  # 0:1:1, skeleton 1, level 2 (P 2)
            (In(1, fluid=0), In(2, fluid=1)),  # 1:2:0, skeleton 0, level 1 (P 3)
            (In(2, fluid=1), In(1, node=0)),  # 0:5:1, skeleton 0, level 1 (P 6)
            (In(1, node=0), In(2, fluid=2)),  # 0:1:5, skeleton 1, level 1 (P 6)
            (In(3, fluid=1), In(2, fluid=2)),  # 0:3:2, skeleton 2, level 1 (P 5)
            (In(1, node=1), In(1, fluid=1), In(1, node=3)),  # 2:11:5
            (In(1, node=2), In(2, fluid=0)),  # 12:5:1
            (In(2, node=4), In(1, fluid=0), In(2, fluid=2)),  # 5:6:14
        ),
        roots=(5, 6, 7),
    )
    comps = graph.compositions()
    assert [comps[r] for r in (5, 6, 7)] == [Composition.from_ints(r, sum(r)) for r in FIG_15]
    assert paper_counts(graph) == (9, 20, 8)
    skel = [
        Node(1, 2, 2, 2),
        Node(0, 1, 3, 3),
        Node(0, 1, 3, 6),
        Node(1, 1, 3, 6),
        Node(2, 1, 5, 5),
        Node(0, 0, 3, 18),
        Node(1, 0, 3, 18),
        Node(2, 0, 5, 25),
    ]
    for d, node in enumerate(graph.nodes):
        for x in node:
            if x.node is not None:
                assert allowed(skel[x.node], skel[d])


def test_fig_15_ours_is_no_worse_than_the_papers() -> None:
    inst = instance(FIG_15, 5)
    graph = mtwm(inst)
    assert paper_counts(graph) == (4, 15, 7)  # paper: 8, 20, 8
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "mtwm"), inst)


def test_table_2_sds_page_12_12_1() -> None:
    """Table 2, R4-R6 at 6%: three targets 12:12:1, M 7; the paper's MTWM: nw 8, nr 23, nm 6."""
    assert paper_counts(mtwm(instance([(12, 12, 1)] * 3, 7))) == (2, 17, 4)


@pytest.mark.parametrize(
    ("ratio", "paper"),
    [((102, 26, 3, 3, 122), (11, 15, 6)), ((180, 26, 26, 5, 5, 14), (11, 15, 6))],
)
def test_stwm_table_1(ratio: tuple[int, ...], paper: tuple[int, int, int]) -> None:
    """Table 1 (M 5): Splinkerette PCR and Silver Restriction Digest; ours is as good or better
    on every count (Splinkerette 7, 11, 5; Silver equal)."""
    ours = paper_counts(stwm(instance([ratio], 5)))
    assert all(o <= p for o, p in zip(ours, paper, strict=True))


def test_one_droplet_per_target() -> None:
    with pytest.raises(NotApplicable):
        mtwm(instance([(12, 12, 1)], 7, demand=2))
    with pytest.raises(NotApplicable):
        stwm(instance(FIG_15, 5))
