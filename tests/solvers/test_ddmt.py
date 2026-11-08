"""MTC (Mitra et al., TCAD 2014) and DDMT (Poddar et al., TODAES 2021) against the papers'
examples, and legal plans (SPEC 10.4). The reading is in `meda.solvers.baselines.ddmt`."""

from __future__ import annotations

from math import ceil

import pytest

from meda.core.composition import ZERO, Composition, Rational
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.ddmt import ddmt, demands, distance, sequence
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from tests.core.helpers import SB, chip


def instance(
    targets: list[tuple[int, int, int]], den: int, tol: Rational = ZERO, limit: int = 1024
) -> Instance:
    ts = tuple(
        Target(i, 0, Composition.from_ints((x, den - x), den), tol, 1, d)
        for i, (x, _, d) in enumerate(targets)
    )
    return Instance(
        "ddmt", SB, ts, chip(max_mix_volume=8, storage_capacity=64, max_denominator=limit), 500
    )


def test_mtc_distances_and_figs_5_6() -> None:
    """10/16 -> 6/16 is 2 steps, back is 3; {5, 11, 14}/16: 6 steps, 4 sample, 3 buffer."""
    assert (distance(10, 6, 4), distance(6, 10, 4)) == (2, 3)
    seq = sequence([5, 11, 14], 4)
    assert len(seq) == 6
    ones = sum(b for _, b in seq)
    assert (ones, len(seq) - ones + 1) == (4, 3)  # the first step also takes a buffer


def test_ddmt_fig_4() -> None:
    seq = sequence([3, 5], 3)
    assert [x for x, _ in seq] == [4, 6, 3, 5]
    assert demands(seq, {3: 4, 5: 8}) == [2, 4, 8, 8]


def test_ddmt_fig_3() -> None:
    """{3, 4, 5}/8 with demands {3, 2, 5}: 10 mix-splits, 7 sample droplets."""
    seq = sequence([3, 4, 5], 3)
    d = demands(seq, {3: 3, 4: 2, 5: 5})
    assert sum(ceil(x / 2) for x in d) == 10
    assert sum(ceil(x / 2) for x, (_, b) in zip(d, seq, strict=True) if b) == 7


@pytest.mark.parametrize(
    ("targets", "den", "tol", "limit"),
    [
        ([(4, 0, 2), (6, 0, 3)], 8, ZERO, 1024),
        ([(13, 0, 1)], 16, ZERO, 1024),
        ([(301, 0, 2)], 512, ZERO, 1024),
        ([(3, 0, 3), (5, 0, 5)], 8, Rational.of(1, 8), 16),
    ],
)
def test_plans_replay_and_validate(
    targets: list[tuple[int, int, int]], den: int, tol: Rational, limit: int
) -> None:
    inst = instance(targets, den, tol, limit)
    _, totals, deltas = replay(inst, emit(inst, ddmt(inst)))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "ddmt"), inst)


def test_a_drifted_exact_target_is_not_applicable() -> None:
    """{5, 11, 14}/16 exact: the path's later droplets are only within 1/16 of their labels."""
    inst = instance([(5, 0, 1), (11, 0, 1), (14, 0, 1)], 16)
    with pytest.raises(NotApplicable):
        emit(inst, ddmt(inst))
