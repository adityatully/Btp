"""SIMOP (Bera et al., SN Computer Science 2021) against the paper's Fig. 1 and Table 5, and
legal plans (SPEC 10.4). The reading is in `meda.solvers.baselines.simop`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from meda.solvers.baselines.simop import chain, counts, simop
from tests.core.helpers import SB, chip


def instance(x: int, den: int, m: int = 8, v: int = 1, demand: int = 1) -> Instance:
    t = Target(0, 0, Composition.from_ints((x, den - x), den), ZERO, v, demand)
    return Instance(
        "simop", SB, (t,), chip(max_mix_volume=m, storage_capacity=64, max_denominator=1024), 500
    )


@pytest.mark.parametrize(
    ("criterion", "expected"), [("buffer", (10, 2, 3, 3)), ("sample", (10, 1, 4, 3))]
)
def test_table_5_313(criterion: str, expected: tuple[int, int, int, int]) -> None:
    """313/1024: buffer first 2 sample, 3 buffer, 3 waste; sample first 1, 4, 3."""
    assert counts(chain(313, 10, criterion)) == expected


def test_fig_1_799() -> None:
    """799/1024 in 10 steps; the paper's graph wastes one droplet each of 672 and 802."""
    assert counts(chain(799, 10, "waste")) == (10, 3, 1, 2)


def test_bit_scanning_is_a_chain() -> None:
    """1/1024 is bit scanning: 10 steps, 1 sample, 10 buffer, 9 waste."""
    assert counts(chain(1, 10, "waste")) == (10, 1, 10, 9)


@pytest.mark.parametrize(("x", "den", "v", "demand"), [(313, 1024, 1, 1), (5, 16, 2, 3)])
def test_plan_replays(x: int, den: int, v: int, demand: int) -> None:
    inst = instance(x, den, v=v, demand=demand)
    graph = simop(inst)
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "simop"), inst)
    if demand == 1:
        c = graph.counts(v)
        drawn = c["drawn"]
        assert isinstance(drawn, tuple)
        assert (c["nm"], drawn[0], drawn[1]) == counts(chain(x, 10))[:3]


def test_not_applicable() -> None:
    with pytest.raises(NotApplicable):
        simop(instance(1, 3))
