"""Bit-scanning: the classic 1:1 binary dilution of the DMFB literature (SPEC 10.4).

Valid only under a 1:1 chip (`dmfb_m2`: `min_mix_volume == max_mix_volume == 2`), two fluids, and
a target whose denominator is a power of two. A target like 3/10 is not dyadic and is genuinely
infeasible here, which is the point of comparing against MEDA's k:l mixing.

Under M = 2 the naive mixed-radix construction (10.4) *is* bit-scanning: every radix is 2, so the
digits are the bits of n and each digit contributes one mix with the sample or the buffer. Rather
than write that twice, this baseline checks its own preconditions and then delegates, and a test
asserts the two agree on every dyadic target. It exists as a separate named baseline because the
literature reports it as one.
"""

from __future__ import annotations

import time

from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.solvers.base import SolveBudget, SolveResult
from meda.solvers.baselines.naive import naive_plan


def is_dyadic(value: int) -> bool:
    return value >= 1 and value & (value - 1) == 0


def applies_to(instance: Instance) -> str | None:
    """None if bit-scanning applies, else why it does not."""
    chip = instance.chip
    if chip.min_mix_volume != 2 or chip.max_mix_volume != 2:
        return (
            f"chip '{chip.name}' is not 1:1 (mix volume "
            f"[{chip.min_mix_volume}, {chip.max_mix_volume}], needs exactly 2)"
        )
    if instance.num_fluids != 2:
        return f"{instance.num_fluids} fluids: bit-scanning dilutes one sample with one buffer"
    for target in instance.targets:
        if not is_dyadic(target.composition.den):
            return (
                f"target {target.id} is {target.composition}, not a dyadic fraction: "
                f"unreachable with 1:1 mixing"
            )
        if target.droplet_volume > 2:
            return f"target {target.id} needs {target.droplet_volume} units, above a 1:1 mix"
    return None


class BitScanSolver:
    """`Solver` for dyadic two-fluid dilutions on a 1:1 chip."""

    name = "bitscan"

    def __init__(self, weights: CostWeights) -> None:
        self.weights = weights

    def solve(self, instance: Instance, budget: SolveBudget | None = None) -> SolveResult:
        start = time.perf_counter()
        reason = applies_to(instance)
        plan = None if reason else naive_plan(instance)
        if plan is None:
            return SolveResult(
                instance.instance_id,
                self.name,
                "infeasible",
                None,
                None,
                None,
                time.perf_counter() - start,
            )
        _, totals, deltas = replay(instance, plan)
        return SolveResult(
            instance_id=instance.instance_id,
            producer=self.name,
            status="feasible",
            dag=MixingDAG.from_deltas(instance, deltas, totals, self.name),
            totals=totals,
            cost=self.weights.scalar(totals, instance.fluids),
            wall_time=time.perf_counter() - start,
        )
