"""MTDA (Kamal et al., TODAES 2020) against the paper's Fig. 5, its Table 4 at two targets, and
legal plans (SPEC 10.4). The reading is in `meda.solvers.baselines.mtda`."""

from __future__ import annotations

import numpy as np
import pytest

from meda.core.composition import ZERO, Composition, Rational
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.mtda import digits_of, mtda
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from tests.core.helpers import SB, chip


def instance(
    xs: list[int], den: int, m: int, v: int = 1, demand: int = 1, tol: Rational = ZERO
) -> Instance:
    ts = tuple(
        Target(i, 0, Composition.from_ints((x, den - x), den), tol, v, demand)
        for i, x in enumerate(xs)
    )
    c = chip(max_mix_volume=m, storage_capacity=256, max_denominator=max(den, 1024))
    return Instance("mtda", SB, ts, c, 100_000)


def counts(inst: Instance, n: int | None = None) -> tuple[int, int, int, int]:
    """(mixes, sample, buffer, waste) of the paper's graph, after a legal replay."""
    graph = mtda(inst, n)
    _, totals, deltas = replay(inst, emit(inst, graph))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "mtda"), inst)
    c = graph.counts(sum(t.demand * t.droplet_volume for t in inst.targets))
    drawn, nw, nm = c["drawn"], c["nw"], c["nm"]
    assert isinstance(drawn, tuple) and isinstance(nw, int) and isinstance(nm, int)
    return nm, drawn[0], drawn[1], nw


def test_fig_5() -> None:
    """90/256 = (0.1122)_4 with mixer-4: root (1 S, (0.1), (0.022), 1 B); (0.022) = (0.2) +
    (0.02) + 2 B; (0.02) = (0.2) + 3 B; (0.2) shared; (0.1) = 1 S + 3 B. Five mixes."""
    inst = instance([90], 256, 4, v=4)
    assert digits_of(inst, 4, 0) == (4, [(1, 1, 2, 2)])
    assert counts(inst, 4) == (5, 4, 11, 11)


def test_table_4_two_targets() -> None:
    """Table 4, N = 2, z = 10, two targets: the paper prints 8.78 mixes, 2.96 sample, 4.85
    buffer, 5.81 waste per target (5000 sets). Ours on 300 sets is within 10% of each."""
    rng = np.random.default_rng(1)
    acc = np.zeros(4)
    for _ in range(300):
        xs = [int(x) for x in rng.choice(np.arange(1, 1024), size=2, replace=False)]
        g = mtda(instance(xs, 1024, 2, v=2), 2)
        c = g.counts(4)
        drawn = c["drawn"]
        assert isinstance(drawn, tuple)
        acc += np.array([c["nm"], drawn[0], drawn[1], c["nw"]], dtype=float) / 2
    ours = acc / 300
    paper = np.array([8.78, 2.96, 4.85, 5.81])
    assert np.all(np.abs(ours / paper - 1) < 0.10), ours


def test_sharing_across_targets() -> None:
    """Two targets with the same prime (0.2)_4 share it: 2 + 1 mixes, not 2 + 2."""
    assert counts(instance([2 * 64 + 2 * 16, 64 + 2 * 16], 256, 4, v=4), 4)[0] == 3


def test_demand_and_small_droplets() -> None:
    """A 1-unit droplet: 4 per root mix of mixer-4, so demand 5 takes 2 root mixes."""
    nm, _, _, _ = counts(instance([1], 4, 4, v=1, demand=5), 4)
    assert nm == 2


def test_picks_n() -> None:
    """Without `segments`, the fewest mixes over every N <= M: 1/2 is one mix at N = 2 or 4."""
    assert counts(instance([2], 4, 4, v=2))[0] == 1


def test_not_applicable() -> None:
    with pytest.raises(NotApplicable):
        mtda(instance([1], 3, 4, v=1), 2)  # 1/3 is no c / 2^z
    with pytest.raises(NotApplicable):
        mtda(instance([1], 4, 4, v=4), 2)  # a 4-unit droplet > mixer-2
    with pytest.raises(NotApplicable):
        mtda(instance([1], 4, 2, v=1), 4)  # mixer-4 > M = 2


def test_tolerance_moves_target() -> None:
    """1/3 within 1/100 is 21/64 at mixer-4 (z = 3)."""
    inst = instance([1], 3, 4, v=4, tol=Rational.of(1, 100))
    z, digits = digits_of(inst, 4, 0)
    assert z == 3 and digits == [(1, 1, 1)]
    counts(inst, 4)


def test_early_outputs_keep_storage_small() -> None:
    """D-97: a target leaves when made and a used-up intermediate is discarded, so 20 targets at
    mixer-4 fit 16 storage cells; holding every target to the end, as the other methods'
    replays do, does not."""
    from dataclasses import replace

    rng = np.random.default_rng(3)
    xs = sorted({int(x) for x in rng.integers(1, 256, size=20)})
    inst = instance(xs, 256, 4, v=4)
    inst = replace(inst, chip=replace(inst.chip, storage_capacity=16))
    graph = mtda(inst, 4)
    replay(inst, emit(inst, graph))
    with pytest.raises(NotApplicable):
        emit(inst, replace(graph, early_outputs=False))
