"""MTCS, MTR and MTMD (Shalu et al., TODAES 2018) against the paper's figures, and legal plans
(SPEC 10.4). The reading of the paper is in `meda.solvers.baselines.mtmd`."""

from __future__ import annotations

import pytest

from meda.core.composition import ZERO, Composition
from meda.core.dag import MixingDAG, validate
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.solvers.baselines.mtmd import clone, mtmd, mtr, paper_counts
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from tests.core.helpers import chip


def fluids(t: int) -> tuple[Fluid, ...]:
    kind = [FluidKind.SAMPLE] + [FluidKind.REAGENT] * (t - 2) + [FluidKind.BUFFER]
    return tuple(Fluid(i, f"R{i + 1}", k) for i, k in enumerate(kind))


def instance(targets: list[tuple[tuple[int, ...], int]], m: int = 8, storage: int = 64):
    ts = tuple(
        Target(i, 0, Composition.from_ints(r, sum(r)), ZERO, 1, d)
        for i, (r, d) in enumerate(targets)
    )
    n = len(targets[0][0])
    return Instance("mtmd", fluids(n), ts, chip(max_mix_volume=m, storage_capacity=storage), 500)


def test_mtcs_fig_3() -> None:
    """7:7:5:5:3:3:2: Tms 9, W 5 (the paper's numbers exactly)."""
    f, roots = mtr([(7, 7, 5, 5, 3, 3, 2)], 7)
    assert paper_counts(f, roots) == (9, 5)


def test_mtr_fig_7() -> None:
    """39:13:12 and 30:23:11: the paper's (random) pairing gives Tms 12, W 8; ours 14, 9."""
    f, roots = mtr([(39, 13, 12), (30, 23, 11)], 3)
    assert paper_counts(f, roots) == (14, 9)


def test_mtmd_fig_12b() -> None:
    """5:3:2:10:12 x 3 and 14:6:5:1:6 x 2: Tms 13, W 7 (the paper's numbers exactly)."""
    f, roots = mtr([(5, 3, 2, 10, 12), (14, 6, 5, 1, 6)], 5)
    for root, demand in zip(roots, (3, 2), strict=True):
        for _ in range(demand - 1):
            clone(f, root)
    assert (len(f.nodes), sum(len(v) for v in f.spare.values())) == (13, 7)


@pytest.mark.parametrize(
    "targets",
    [
        [((7, 7, 5, 5, 3, 3, 2), 1)],
        [((5, 3, 2, 10, 12), 3), ((14, 6, 5, 1, 6), 2)],
        [((39, 13, 12), 2), ((30, 23, 11), 4)],
    ],
)
def test_plans_replay_and_validate(targets: list[tuple[tuple[int, ...], int]]) -> None:
    inst = instance(targets)
    _, totals, deltas = replay(inst, emit(inst, mtmd(inst)))
    validate(MixingDAG.from_deltas(inst, deltas, totals, "mtmd"), inst)


def test_a_non_dyadic_exact_target_is_not_applicable() -> None:
    with pytest.raises(NotApplicable):
        mtmd(instance([((1, 1, 1), 1)]))
