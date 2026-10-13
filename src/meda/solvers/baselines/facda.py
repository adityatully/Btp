"""FacDA: factorisation-based MEDA dilution (ASP-DAC 2019) — NOT IMPLEMENTED (SPEC 10.4).

Reserved for M9. The acceptance rule from the research notes is strict: **an implemented baseline
must reproduce its published numbers inside our simulator before any agent is trained on
comparisons against it.** If it cannot, the simulator is wrong, and that is the finding.

So this stays a stub rather than a plausible-looking approximation that would quietly produce
numbers nobody has checked. Two things must be settled before writing it:

- the author list is unconfirmed (Kundu et al. vs Bhattacharjee et al.) — check the PDF;
- the paper's reported results must be transcribed into `literature_v1` first, so the
  reproduction test has something to compare against.

Reported in the paper (to be verified, not yet relied on): ~38% fewer steps, 52% less sample and
23% less waste than earlier MEDA methods, for M in 4..10.
"""

from __future__ import annotations

from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.solvers.base import SolveBudget, SolveResult


class FacDASolver:
    """Stub. `verified` stays False until it reproduces the paper on `literature_v1`."""

    name = "facda"
    verified = False

    def __init__(self, weights: CostWeights) -> None:
        self.weights = weights

    def solve(self, instance: Instance, budget: SolveBudget | None = None) -> SolveResult:
        raise NotImplementedError(
            "FacDA is not implemented (SPEC 10.4, reserved for M9). It must reproduce the "
            "paper's reported numbers on the literature_v1 benchmark before it is used as a "
            "baseline; a guessed implementation would produce unchecked numbers."
        )
