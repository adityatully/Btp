"""Dong's splitting-droplet sharing (JETTA 2024) against the paper's Figs. 8 and 10, and legal
plans (SPEC 10.4). The reading is in `meda.solvers.baselines.dong`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition, Rational
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.dong import dong
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from tests.core.helpers import SB, chip


def instance(
    targets: list[tuple[int, int]], den: int, m: int, v: int = 1, tol: Rational = ZERO
) -> Instance:
    ts = tuple(
        Target(i, 0, Composition.from_ints((x, den - x), den), tol, v, d)
        for i, (x, d) in enumerate(targets)
    )
    return Instance(
        "dong", SB, ts, chip(max_mix_volume=m, storage_capacity=64, max_denominator=1024), 500
    )


def counts(inst: Instance) -> tuple[int, int, int, int]:
    """(mixes, sample units, buffer units, waste units) of a replayed plan."""
    graph = dong(inst)
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "dong"), inst)
    c = graph.counts(sum(t.demand * t.droplet_volume for t in inst.targets))
    drawn, nw, nm = c["drawn"], c["nw"], c["nm"]
    assert isinstance(drawn, tuple) and isinstance(nw, int) and isinstance(nm, int)
    return nm, drawn[0], drawn[1], nw


def test_fig_10() -> None:
    """848 / 1024 in a 1024-cell droplet: one mix of 848 sample and 176 buffer cells."""
    assert counts(instance([(848, 1)], 1024, 1024, v=1024)) == (1, 848, 176, 0)


def test_fig_8_gradient() -> None:
    """The gradient's sample is the sum of its pieces (the 1 bits): 1408 cells, no waste."""
    ts = [(x, 1) for x in (64, 160, 208, 232, 244, 248, 252)]
    nm, sample, _, nw = counts(instance(ts, 1024, 1024, v=1024))
    assert (nm, sample, nw) == (7, 1408, 0)


def test_demand_shares_a_mix() -> None:
    """3/4 at M 8, three 2-unit droplets: one 8-unit mix (q = 4 fits) gives three, 2 left."""
    assert counts(instance([(3, 3)], 4, 8, v=2)) == (1, 6, 2, 2)


def test_tolerant_target_moves_to_a_dyadic_cf() -> None:
    """1/3 within 1/12 is 1/4 at M 4."""
    assert counts(instance([(1, 1)], 3, 4, tol=Rational.of(1, 12))) == (1, 1, 3, 3)


@pytest.mark.parametrize(
    ("targets", "den", "m"), [([(1, 1)], 3, 8), ([(5, 1)], 16, 8)], ids=["not dyadic", "above M"]
)
def test_not_applicable(targets: list[tuple[int, int]], den: int, m: int) -> None:
    with pytest.raises(NotApplicable):
        dong(instance(targets, den, m))
