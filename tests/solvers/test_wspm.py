"""WSPM (Li et al., TBioCAS 2017) against the paper's Figs. 6, 10 and 12, and legal plans
(SPEC 10.4). The reading is in `meda.solvers.baselines.wspm`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.paper_graph import emit
from meda.solvers.baselines.wspm import wspm
from tests.core.helpers import SB, chip


def instance(targets: list[tuple[int, int]], den: int, m: int) -> Instance:
    ts = tuple(
        Target(i, 0, Composition.from_ints((x, den - x), den), ZERO, 1, d)
        for i, (x, d) in enumerate(targets)
    )
    return Instance(
        "wspm", SB, ts, chip(max_mix_volume=m, storage_capacity=64, max_denominator=1024), 500
    )


def counts(inst: Instance) -> tuple[int, int, int, int]:
    """(operations, sample units, buffer units, waste units) of a replayed plan."""
    graph = wspm(inst)
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "wspm"), inst)
    c = graph.counts(sum(t.demand for t in inst.targets))
    drawn, nw, nm = c["drawn"], c["nw"], c["nm"]
    assert isinstance(drawn, tuple) and isinstance(nw, int) and isinstance(nm, int)
    return nm, drawn[0], drawn[1], nw


def test_fig_6c() -> None:
    """1/3 with 1X and 2X droplets: one operation, 64(1) with 0(2), 1 sample, 2 waste."""
    assert counts(instance([(1, 1)], 3, 3)) == (1, 1, 2, 2)


def test_fig_12a_first_target() -> None:
    """80/256 with 1X droplets: 4 operations from one sample droplet (Fig. 10's PD sharing)."""
    nm, sample, _, _ = counts(instance([(80, 1)], 256, 2))
    assert (nm, sample) == (4, 1)


def test_fig_12d_incremental() -> None:
    """80 then 56 at 1/256: the paper's incremental WSPM uses 1 sample, 3 buffer, 5 operations
    (its 4 waste droplets count reused ones; ours counts 2 units left at the end)."""
    assert counts(instance([(80, 1), (56, 1)], 256, 2)) == (5, 1, 3, 2)


@pytest.mark.parametrize(("targets", "den", "m"), [([(301, 2)], 512, 8), ([(5, 1), (3, 3)], 16, 4)])
def test_plans_replay_and_validate(targets: list[tuple[int, int]], den: int, m: int) -> None:
    counts(instance(targets, den, m))
