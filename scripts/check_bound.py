"""Check the exact-search lower bound against proved optima (SPEC 10.2, 0.21).

    uv run python scripts/check_bound.py --workers 32

Draws small random problems (dilutions and 3-fluid mixtures, exact and tolerant targets,
1-2 targets with demand 1-2, M 3-5, storage 2, 3 or 16), proves each optimum with plain
Dijkstra (no heuristic) where the node budget allows, and checks two things on every proved
problem:

1. the bound at the root never exceeds the optimum (admissible);
2. A* with the bound finds the same optimum (the bound prunes nothing it should not).

Problems Dijkstra cannot prove within the budget are counted and skipped. Exits 1 if any check
fails, naming the problems. Each problem takes seconds to minutes, so run it with many workers
on a machine with cores to spare, not a laptop.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from meda.core.chip import ChipLimits
from meda.core.composition import ZERO, Composition, Rational
from meda.core.costs import CostWeights
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.state import PrepState
from meda.solvers.base import SolveBudget
from meda.solvers.exact_search import ExactSearch, fluid_bound_heuristic

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SB = (Fluid(0, "S", FluidKind.SAMPLE), Fluid(1, "B", FluidKind.BUFFER))
SRB = (
    Fluid(0, "S", FluidKind.SAMPLE),
    Fluid(1, "R", FluidKind.REAGENT),
    Fluid(2, "B", FluidKind.BUFFER),
)
STEP_LIMIT = 14


def weights() -> CostWeights:
    return CostWeights.from_dict(
        OmegaConf.to_container(OmegaConf.load(CONFIGS / "costs/geometric.yaml"))
    )  # type: ignore[arg-type]


def draw(count: int, seed: int) -> list[Instance]:
    """The same `count` problems for a given seed, whatever the worker count."""
    rng = np.random.default_rng(seed)
    base = ChipLimits.from_dict(
        OmegaConf.to_container(OmegaConf.load(CONFIGS / "chip/meda_m8.yaml"))
    )  # type: ignore[arg-type]
    out = []
    for i in range(count):
        m = int(rng.integers(3, 6))
        mixture = rng.random() < 0.4
        tol = [ZERO, Rational.of(1, 20), Rational.of(1, 10)][int(rng.integers(3))]
        targets = []
        for t in range(int(rng.integers(1, 3))):
            den = int(rng.choice([2, 3, 4, 5, 6]))
            a = int(rng.integers(1, den))
            if mixture:
                b = int(rng.integers(0, den - a + 1))
                nums: tuple[int, ...] = (a, b, den - a - b)
            else:
                nums = (a, den - a)
            volume, demand = int(rng.integers(1, m)), int(rng.integers(1, 3))
            targets.append(Target(t, t, Composition.from_ints(nums, den), tol, volume, demand))
        cap = int(rng.choice([2, 3, 16]))
        chip = replace(base, max_mix_volume=m, storage_capacity=cap)
        out.append(Instance(f"rand{i}", SRB if mixture else SB, tuple(targets), chip, STEP_LIMIT))
    return out


def check(job: tuple[Instance, int]) -> tuple[str, str | None, str, str, str]:
    """(id, proved optimum or None, root bound, A* cost, A* status), costs as exact strings."""
    inst, max_nodes = job
    w = weights()
    budget = SolveBudget(max_nodes=max_nodes)
    plain = ExactSearch(w, heuristic="zero", use_incumbent=False).solve(inst, budget)
    if plain.status != "optimal":
        return inst.instance_id, None, "", "", ""
    bound = fluid_bound_heuristic(w)(PrepState.initial(inst), inst)
    astar = ExactSearch(w, heuristic="fluid_bound", use_incumbent=False).solve(inst, budget)
    return inst.instance_id, str(plain.cost), str(bound), str(astar.cost), astar.status


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--problems", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-nodes", type=int, default=60_000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    start = time.time()
    jobs = [(inst, args.max_nodes) for inst in draw(args.problems, args.seed)]
    proved, bad, ratios = 0, [], []
    with Pool(args.workers) as pool:
        for done, (iid, opt, bound, astar, status) in enumerate(
            pool.imap_unordered(check, jobs), 1
        ):
            if opt is not None:
                proved += 1
                o, b = Fraction(opt), Fraction(bound)
                ratios.append(float(b / o))
                if b > o:
                    bad.append(f"{iid}: bound {bound} > optimum {opt}")
                if astar != opt:
                    bad.append(f"{iid}: A* found {astar} ({status}), optimum {opt}")
            if done % 25 == 0 or done == len(jobs):
                print(
                    f"{done}/{len(jobs)} checked, {proved} proved, {len(bad)} failures, "
                    f"{time.time() - start:.0f} s",
                    flush=True,
                )

    print(f"\nproblems {len(jobs)}, optimum proved on {proved}, failures {len(bad)}")
    if ratios:
        print(
            f"bound / optimum: median {statistics.median(ratios):.3f}, "
            f"min {min(ratios):.3f}, max {max(ratios):.3f}"
        )
    for line in bad:
        print("FAIL", line)
    print("PASS" if not bad else "FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
