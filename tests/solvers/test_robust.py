"""Robust split-less dilution (Poddar et al., TODAES 2020) against the paper's Example 5, and
legal plans for many targets with demand (SPEC 10.4). The reading is in
`meda.solvers.baselines.robust`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from meda.solvers.baselines.robust import least, robust
from tests.core.helpers import SB, chip


def instance(targets: list[tuple[int, int, int]], den: int, m: int = 8) -> Instance:
    ts = tuple(
        Target(i, 0, Composition.from_ints((x, den - x), den), ZERO, v, d)
        for i, (x, v, d) in enumerate(targets)
    )
    return Instance(
        "robust", SB, ts, chip(max_mix_volume=m, storage_capacity=64, max_denominator=den), 500
    )


def test_example_5_aliquots() -> None:
    """71/128 at n 7: ECFs {2, 8, 32, 128}/128 in 1, 1, 1, 3 aliquots, a droplet of 6."""
    assert least(71, 7, 1, 64) == (6, [7, 7, 7, 5, 3, 1])


@pytest.mark.parametrize(
    ("targets", "den"),
    [
        ([(71, 1, 1)], 128),
        ([(3, 1, 2), (5, 1, 1)], 8),
        ([(2, 1, 1), (7, 1, 3)], 8),
        ([(13, 1, 1), (5, 2, 2)], 16),
        ([(301, 1, 1)], 512),
    ],
)
def test_plans_replay_and_validate(targets: list[tuple[int, int, int]], den: int) -> None:
    inst = instance(targets, den)
    _, totals, deltas = replay(inst, emit(inst, robust(inst)))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "robust"), inst)


def test_a_non_dyadic_exact_target_is_not_applicable() -> None:
    with pytest.raises(NotApplicable):
        robust(instance([(1, 1, 1)], 3))
