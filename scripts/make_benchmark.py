"""Build and freeze a versioned benchmark set (SPEC 8.4).

    uv run python scripts/make_benchmark.py tiny_exact_v1
    uv run python scripts/make_benchmark.py val_stage1_v1
    uv run python scripts/make_benchmark.py val_stage1_v2
    uv run python scripts/make_benchmark.py val_s1_v2      # one per stage of track2_v1 (8.3)
    uv run python scripts/make_benchmark.py heldout_combo_v2
    uv run python scripts/make_benchmark.py --list

A published version is frozen: the script refuses to overwrite one unless `--overwrite` is
given, and the right move after a definition changes is to bump the version. Chip and cost
configs are read straight from `configs/`, because a set may mix chips (the hand-checked cases
of SPEC 17.3 use M = 2, M = 4 and M = 8).

`--store-optima` runs the exact search over an existing set and rewrites its `reference` with
proved optima (status `optimal`). The instances themselves are untouched.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Callable
from dataclasses import replace
from math import ceil, lcm
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from meda.core.chip import ChipLimits
from meda.core.composition import ZERO, Composition, Rational
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG, validate
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet, InstanceReference, instance_hash
from meda.generators.sampler import (
    GeneratorConfig,
    GeneratorSource,
    dilution_grid,
    sample_instances,
    step_limit_base,
)
from meda.generators.stages import Cell, StageDesign
from meda.solvers.base import SolveBudget
from meda.solvers.baselines.naive import naive_plan
from meda.solvers.baselines.tree import tree_plan
from meda.solvers.exact_search import ExactSearch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SB = (Fluid(0, "S", FluidKind.SAMPLE), Fluid(1, "B", FluidKind.BUFFER))


def load_chip(name: str) -> ChipLimits:
    data = OmegaConf.to_container(OmegaConf.load(CONFIGS / "chip" / f"{name}.yaml"))
    assert isinstance(data, dict)
    return ChipLimits.from_dict(data)  # type: ignore[arg-type]


def load_weights(name: str = "default") -> CostWeights:
    data = OmegaConf.to_container(OmegaConf.load(CONFIGS / "costs" / f"{name}.yaml"))
    assert isinstance(data, dict)
    return CostWeights.from_dict(data)  # type: ignore[arg-type]


def load_design(name: str) -> StageDesign:
    data = OmegaConf.to_container(OmegaConf.load(CONFIGS / "stages" / f"{name}.yaml"))
    assert isinstance(data, dict)
    return StageDesign.from_dict(data)  # type: ignore[arg-type]


def load_generator(name: str) -> tuple[GeneratorConfig, str]:
    """A generator stage plus the chip it overrides, read from its own YAML."""
    raw = OmegaConf.load(CONFIGS / "generator" / f"{name}.yaml")
    data = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(data, dict)
    chip = "meda_m8"
    for entry in data.pop("defaults", []) or []:
        if isinstance(entry, dict):
            for key, value in entry.items():
                if "chip" in str(key):
                    chip = str(value)
    return GeneratorConfig.from_dict(data), chip  # type: ignore[arg-type]


def at_accuracy(chip: ChipLimits, targets: tuple[Target, ...]) -> ChipLimits:
    """The chip at the accuracy level these targets need: `max_denominator` = lcm of their
    denominators.

    This is what makes a set exactly searchable, and it is the only deviation these instances
    make from the chip configs. At `max_denominator: 256` the search wanders through 129
    compositions no target needs and does not finish; at the matched level every instance of
    `tiny_exact_v1` solves in seconds. It is also how the DMFB literature defines things: an
    accuracy level of d means denominators of exactly 2^d, not "up to 256".
    """
    return replace(chip, max_denominator=lcm(*[t.composition.den for t in targets]))


def dilution(
    instance_id: str,
    shares: list[tuple[int, int]],
    chip: ChipLimits,
    demand: int = 1,
    droplet_volume: int = 1,
    note: str = "",
) -> Instance:
    """A sample+buffer instance at its own accuracy level, with a step limit twice its naive
    witness plan."""
    targets = tuple(
        Target(i, 0, Composition.from_ints((n, d - n), d), ZERO, droplet_volume, demand)
        for i, (n, d) in enumerate(shares)
    )
    chip = at_accuracy(chip, targets)
    probe = Instance(instance_id, SB, targets, chip, 10_000)
    actions = naive_plan(probe)
    if actions is None:
        raise ValueError(f"instance '{instance_id}': the naive construction found no plan")
    _, totals, _ = replay(probe, actions)
    metadata: dict[str, Any] = {
        "generator": "hand",
        "note": note,
        "witness_actions": len(actions),
        "known_upper_bound": str(load_weights().scalar(totals, SB)),
        "witness_source": "naive",
    }
    return Instance(instance_id, SB, targets, chip, 2 * len(actions), metadata)


def build_tiny_exact() -> BenchmarkSet:
    """Small enough for exact search: the four hand-checked cases plus every n/8 dilution.

    The hand-checked cases (SPEC 17.3) carry the optima the exact search must reproduce, so
    M5's two acceptance criteria become one check.
    """
    m2, m4, m8 = load_chip("dmfb_m2"), load_chip("meda_m4"), load_chip("meda_m8")
    instances = [
        dilution("tiny-quarter-m2", [(1, 4)], m2, note="SPEC 17.3 case 1: optimum 2 mixes"),
        dilution("tiny-quarter-m4", [(1, 4)], m4, note="SPEC 17.3 case 2: optimum 1 mix"),
        dilution(
            "tiny-three-tenths-m8",
            [(3, 10)],
            m8,
            note="SPEC 17.3 case 3: optimum 2 mixes; infeasible under dmfb_m2",
        ),
        dilution(
            "tiny-sharing-m2",
            [(3, 4), (1, 4)],
            m2,
            note="SPEC 17.3 case 4: optimum 3 mixes, naive 4",
        ),
    ]
    # Every distinct dilution n/8 under M = 2: the full sweep at a tiny accuracy level.
    for inst in dilution_grid(m2, 8, step_limit=10_000):
        shares = [(inst.targets[0].composition.num[0], inst.targets[0].composition.den)]
        name = f"tiny-grid-m2-{shares[0][0]}of{shares[0][1]}"
        candidate = dilution(name, shares, m2, note="dilution sweep at denominator 8")
        if instance_hash(candidate) not in {instance_hash(i) for i in instances}:
            instances.append(candidate)
    return BenchmarkSet(
        name="tiny_exact",
        version=1,
        instances=tuple(instances),
        generator={
            "builder": "scripts/make_benchmark.py:build_tiny_exact",
            "chips": ["dmfb_m2", "meda_m4", "meda_m8"],
            "accuracy": "max_denominator per instance = lcm of its target denominators",
            "contents": "SPEC 17.3 hand-checked cases + every distinct dilution n/8 at M = 2",
        },
        description=(
            "Instances small enough for exact search: each runs its chip at the accuracy level "
            "its own targets need, which is what lets the search finish. Optima are proved by "
            "the exact search and stored in `reference` (--store-optima)."
        ),
    )


def build_val_stage1() -> BenchmarkSet:
    """In-distribution validation for curriculum stage 1 (`tiny_dilution`).

    Stage 1 has only 31 distinct problems (the reduced fractions n/32), so this set is the part
    of the stage that is *not* in `tiny_exact_v1`. Training must pass these hashes to the
    sampler as `exclude`.
    """
    cfg, chip_name = load_generator("tiny_dilution")
    chip = load_chip(chip_name)
    weights = load_weights()
    held_out = BenchmarkSet.read("tiny_exact_v1").hashes()
    seeds = (20261010,)
    rng = np.random.default_rng(seeds[0])
    instances = sample_instances(cfg, chip, rng, count=8, weights=weights, exclude=held_out)
    return BenchmarkSet(
        name="val_stage1",
        version=1,
        instances=tuple(instances),
        generator={
            "builder": "scripts/make_benchmark.py:build_val_stage1",
            "stage": cfg.name,
            "chip": chip_name,
            "config_hash": cfg.config_hash(),
            "excludes": "tiny_exact_v1",
        },
        seeds=seeds,
        description=(
            "Stage-1 validation: 8 dilutions drawn from tiny_dilution, disjoint from "
            "tiny_exact_v1. Stage 1 holds only 31 distinct problems, so training must exclude "
            "these hashes."
        ),
    )


# val_stage1_v2 search budgets, in expanded nodes rather than seconds, so the answer key
# rebuilds identically on any machine.
V2_NODES_REAL = 100_000
V2_NODES_MATCHED = 50_000


def build_val_stage1_v2() -> BenchmarkSet:
    """`val_stage1_v1`'s instances, unchanged, with a better answer key (SPEC 8.4).

    Per instance: if exact search proves the optimum at the instance's own accuracy within
    `V2_NODES_REAL`, store it as `optimal`. Otherwise store the interval [lower bound, best
    known]: the bound is the search's proved `lower_bound` at its budget stop, and the best
    known is the cheapest of the naive plan, the search's own best plan, and a plan found at the
    matched accuracy level (`max_denominator` = the target's denominator). That last plan is
    legal on the real instance too, since its denominators stay within the chip's; it is
    replayed and validated there before it counts.
    """
    v1 = BenchmarkSet.read("val_stage1_v1")
    weights = load_weights()
    solver = ExactSearch(weights)
    reference: dict[str, InstanceReference] = {}
    for inst in v1.instances:
        iid = inst.instance_id
        real = solver.solve(inst, SolveBudget(max_nodes=V2_NODES_REAL))
        if real.status == "optimal":
            assert real.cost is not None and real.totals is not None
            reference[iid] = InstanceReference(
                real.cost,
                "optimal",
                solver.name,
                {"mix_ops": real.totals.mix_ops, "nodes_expanded": real.nodes_expanded},
            )
            print(f"  {iid}: optimal {real.cost}", flush=True)
            continue
        if real.status != "timeout" or real.lower_bound is None:
            raise RuntimeError(f"instance '{iid}': exact search returned {real.status}")
        candidates = [(v1.reference[iid].cost, "naive")]
        if real.cost is not None:
            candidates.append((real.cost, "exact_search (best plan at budget stop)"))
        matched = replace(inst, chip=at_accuracy(inst.chip, inst.targets))
        plan, _, mstats = solver.search(matched, SolveBudget(max_nodes=V2_NODES_MATCHED))
        if plan is not None:
            final, totals, deltas = replay(inst, plan)
            if not final.is_success:
                raise RuntimeError(
                    f"instance '{iid}': matched-accuracy plan fails on the real chip"
                )
            dag = MixingDAG.from_deltas(inst, deltas, totals, solver.name)
            validate(dag, inst).raise_if_invalid()
            den = matched.chip.max_denominator
            status = "best plan at budget stop" if mstats.stopped else "optimal there"
            candidates.append(
                (
                    weights.scalar(totals, inst.fluids),
                    f"exact_search at max_denominator {den} ({status})",
                )
            )
        cost, producer = min(candidates, key=lambda c: c[0])
        reference[iid] = InstanceReference(
            cost,
            "feasible",
            producer,
            {
                "lower_bound_producer": f"exact_search fluid_bound, {V2_NODES_REAL} nodes",
                "nodes_expanded": real.nodes_expanded,
                "naive_cost": str(v1.reference[iid].cost),
            },
            lower_bound=real.lower_bound,
        )
        print(f"  {iid}: [{real.lower_bound}, {cost}] best known from {producer}", flush=True)
    return replace(
        v1,
        version=2,
        generator={**v1.generator, "answer_key": "scripts/make_benchmark.py:build_val_stage1_v2"},
        description=(
            v1.description + " v2 keeps the instances and replaces the answer key: proved optima "
            "where exact search finishes, else [proved lower bound, best known]."
        ),
        reference=reference,
    )


def build_design_set(
    design_name: str,
    which: int | None,
    full: str | None = None,
    seed: int | None = None,
    size: int | None = None,
    exclude: frozenset[str] = frozenset(),
) -> BenchmarkSet:
    """A validation set of a stage design (SPEC 8.3, 8.4): stage `which`'s cells, or the
    held-out cells when `which` is None. `val_size` instances split evenly over the cells,
    each a fresh draw (unwitnessed draws are redrawn), no two the same problem.

    `full`, `seed`, `size` and `exclude` override the set's name, seed, size and the problems
    it may not contain (the literature benchmark draws its own sets this way, plan 2.4).

    Answer key per instance: [lower bound, best known], where best known is the cheaper of the
    naive and split-tree plans and the lower bound is exact search's conservation bound at the
    root (`max_nodes=0`; fluid, waste and mix terms since SPEC 0.21, hence the `_v2` sets).
    Measured on the design's ranges, exact search proved none of 24 stage 1-3 draws within
    1,000 nodes (1-100 s each), so a node budget buys nothing here;
    when the two ends meet the construction is proved optimal and stored as `optimal`.
    """
    design = load_design(design_name)
    chip = load_chip(design.chip)
    weights = load_weights(design.costs)
    if which is None:
        cells: tuple[Cell, ...] = design.held_out
        label, offset = "held-out cells", len(design.stages)
        full = full or design.held_out_val
    else:
        stage = design.stages[which]
        cells, default, label, offset = stage.cells, stage.val, f"stage {stage.name}", which
        full = full or default
    name, version = full.rsplit("_v", 1)
    seeds = (design.val_seed + offset if seed is None else seed,)
    rng = np.random.default_rng(seeds[0])
    per_cell = ceil((size or design.val_size) / len(cells))
    seen: set[str] = set(exclude)
    instances: list[Instance] = []
    for cell in cells:
        drawn = 0
        while drawn < per_cell:
            inst = GeneratorSource(cell.config, chip, weights, frozenset(seen))(rng)
            seen.add(instance_hash(inst))
            instances.append(
                replace(
                    inst,
                    instance_id=f"{name}-{len(instances):03d}",
                    metadata={**inst.metadata, "cell": cell.name},
                )
            )
            drawn += 1
    reference = answer_key(instances, weights)
    return BenchmarkSet(
        name=name,
        version=int(version),
        instances=tuple(instances),
        reference=reference,
        generator={
            "builder": "scripts/make_benchmark.py:build_design_set",
            "design": design.name,
            "which": label,
            "cells": {c.name: c.config.config_hash() for c in cells},
            "chip": design.chip,
            "costs": design.costs,
            "answer_key": "[root conservation bound (SPEC 0.21), cheaper of naive and tree]",
        },
        seeds=seeds,
        description=(
            f"{'Validation' if size is None else 'Literature benchmark'} for {label} of "
            f"stage design {design.name}: {per_cell} draws per cell "
            f"over {', '.join(c.name for c in cells)}. Graded under costs/{design.costs}."
        ),
    )


def answer_key(instances: list[Instance], weights: CostWeights) -> dict[str, InstanceReference]:
    """[root conservation bound, cheaper of naive and tree] per instance (SPEC 0.21)."""
    solver = ExactSearch(weights)
    reference: dict[str, InstanceReference] = {}
    for inst in instances:
        result = solver.solve(inst, SolveBudget(max_nodes=0))
        if result.cost is None or result.lower_bound is None:
            raise RuntimeError(f"instance '{inst.instance_id}': no construction plan")
        proved = result.cost == result.lower_bound
        reference[inst.instance_id] = InstanceReference(
            result.cost,
            "optimal" if proved else "feasible",
            "cheapest_construction",
            {
                "lower_bound_producer": "exact_search fluid_bound at the root (SPEC 0.21)",
                "mix_ops": result.totals.mix_ops if result.totals else None,
            },
            lower_bound=None if proved else result.lower_bound,
        )
    return reference


# The literature benchmark's problems from the papers' own examples and tables (plan 2.4):
# (stage, ratio, denominator, tolerance, M, droplet volume, source). A droplet volume the
# paper states is kept (Poddar et al. prepare 4X and 8X targets); otherwise one unit.
PAPER_PROBLEMS: list[tuple[str, tuple[int, ...], int, Rational, int, int, str]] = [
    ("s1", (70, 49), 119, Rational.of(1, 200), 6, 1, "FacDA (TCAD 2022) Example 2, Fig. 6-8"),
    ("s1", (7, 20), 27, Rational.of(1, 200), 4, 1, "FacDA (TCAD 2022) Fig. 2, 4"),
    ("s1", (7, 5), 12, ZERO, 4, 1, "Generic (TCAD 2022) Fig. 6(b)"),
    *[
        (
            "s1",
            (x, 64 - x),
            64,
            ZERO,
            m,
            m,
            f"Generic (TCAD 2022) Table II(b), {m}X"
            + (", Table III" if x in (1, 17, 31, 44, 45, 56, 61, 63) and m == 4 else "")
            + (", Example 3" if x == 42 and m == 4 else ""),
        )
        for m in (4, 8)
        for x in range(1, 64)
    ],
    ("s2", (7, 7, 6), 20, ZERO, 5, 1, "DFMM/STWM (DSD 2022) Fig. 3, 5"),
    ("s2", (25, 31, 44), 100, ZERO, 8, 4, "MRCM (ASP-DAC 2019) Fig. 3"),
    ("s2", (1, 1, 1, 1), 4, ZERO, 8, 4, "MRCM (ASP-DAC 2019) Section 2.2"),
]


def build_litbench_paper() -> BenchmarkSet:
    """One-target stage-1/2 problems from the chosen papers' worked examples and tables, in
    our format (plan 2.4). Ratios are reduced, as every target of ours is; the papers' tables
    sometimes work on the unreduced x/64."""
    weights = load_weights("geometric")
    base = load_chip("meda_m8")
    kinds = {2: SB, 3: (Fluid(0, "S", FluidKind.SAMPLE), Fluid(1, "R1", FluidKind.REAGENT),
                        Fluid(2, "B", FluidKind.BUFFER))}  # fmt: skip
    kinds[4] = (*kinds[3][:2], Fluid(2, "R2", FluidKind.REAGENT), Fluid(3, "B", FluidKind.BUFFER))
    instances: list[Instance] = []
    for stage, ratio, den, tol, m, volume, source in PAPER_PROBLEMS:
        chip = replace(base, name=f"{base.name}-M{m}-S32", max_mix_volume=m,
                       storage_capacity=32, max_denominator=1024)  # fmt: skip
        target = Target(0, 0, Composition.from_ints(ratio, den), tol, volume, 1)
        probe = Instance("probe", kinds[len(ratio)], (target,), chip, 10_000)
        witness = tree_plan(probe)
        if witness is None:
            raise ValueError(f"{source}: the split tree finds no plan for {ratio}/{den}")
        limit = max(4 * step_limit_base(probe, "depth_support"), 2 * len(witness))
        instances.append(
            replace(
                probe,
                instance_id=f"litbench_paper-{len(instances):03d}",
                step_limit=limit,
                metadata={
                    "generator": "paper",
                    "source": source,
                    "stage": stage,
                    "witness_actions": len(witness),
                    "witness_source": "tree",
                },
            )
        )
    return BenchmarkSet(
        name="litbench_paper",
        version=1,
        instances=tuple(instances),
        reference=answer_key(instances, weights),
        generator={
            "builder": "scripts/make_benchmark.py:build_litbench_paper",
            "costs": "geometric",
            "answer_key": "[root conservation bound (SPEC 0.21), cheaper of naive and tree]",
        },
        seeds=(),
        description=(
            "Literature benchmark (plan 2.4): one-target dilution (s1) and mixture (s2) problems "
            "from the worked examples and tables of FacDA, Generic, DFMM and MRCM. Graded under "
            "costs/geometric."
        ),
    )


def design_builders(design_name: str) -> dict[str, Callable[[], BenchmarkSet]]:
    design = load_design(design_name)
    out: dict[str, Callable[[], BenchmarkSet]] = {
        s.val: functools.partial(build_design_set, design_name, i)
        for i, s in enumerate(design.stages)
    }
    out[design.held_out_val] = functools.partial(build_design_set, design_name, None)
    return out


def build_litbench(which: int) -> BenchmarkSet:
    """Fresh draws of track2_v1 stage `which` for the literature benchmark (plan 2.4): 100
    problems on a new seed, none the same problem as any set already in `benchmarks/`."""
    paths = sorted((ROOT / "benchmarks").glob("*_v*.json"))
    held = frozenset().union(
        *(BenchmarkSet.read(p.stem).hashes() for p in paths if not p.stem.startswith("litbench"))
    )
    return build_design_set(
        "track2_v1", which, f"litbench_s{which + 1}_v1", 20261027 + which, 100, held
    )


BUILDERS: dict[str, Callable[[], BenchmarkSet]] = {
    "tiny_exact_v1": build_tiny_exact,
    "val_stage1_v1": build_val_stage1,
    "val_stage1_v2": build_val_stage1_v2,
    **design_builders("track2_v1"),
    "litbench_s1_v1": functools.partial(build_litbench, 0),
    "litbench_s2_v1": functools.partial(build_litbench, 1),
    "litbench_paper_v1": build_litbench_paper,
}


def store_optima(name: str, out: Path, seconds: float, overwrite: bool) -> None:
    """Prove and store the optimum of every instance in a set (SPEC 10.2, M5 acceptance)."""
    bench = BenchmarkSet.read(name, out)
    solver = ExactSearch(load_weights())
    reference: dict[str, InstanceReference] = {}
    for inst in bench.instances:
        result = solver.solve(inst, SolveBudget(max_seconds=seconds))
        stats = solver.last_stats
        if result.status != "optimal" or result.cost is None:
            raise RuntimeError(
                f"instance '{inst.instance_id}': exact search returned {result.status} after "
                f"{result.wall_time:.1f}s. A set named *_exact must be searchable; lower its "
                f"accuracy level or shrink the instance."
            )
        reference[inst.instance_id] = InstanceReference(
            cost=result.cost,
            status="optimal",
            producer=solver.name,
            detail={
                "mix_ops": result.totals.mix_ops if result.totals else None,
                "nodes_expanded": result.nodes_expanded,
                "search_exhausted": bool(stats and stats.exhausted),
            },
        )
        print(
            f"  {inst.instance_id:24s} optimum {result.cost!s:>7s} "
            f"({result.totals.mix_ops if result.totals else '?'} mixes, "
            f"{result.wall_time:.2f}s)"
        )
    path = bench.with_reference(reference).write(out, overwrite=True)
    print(f"stored {len(reference)} optima in {path}")


def summarise(bench: BenchmarkSet) -> str:
    lines = [f"{bench.full_name}: {len(bench)} instances"]
    for inst in bench.instances:
        targets = ", ".join(f"{t.composition}x{t.demand}" for t in inst.targets)
        bound = inst.metadata.get("known_upper_bound", "?")
        lines.append(
            f"  {inst.instance_id:28s} chip={inst.chip.name:9s} steps<={inst.step_limit:5d} "
            f"upper_bound={bound:>8s}  {targets}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("set", nargs="?", choices=sorted(BUILDERS), help="which set to build")
    parser.add_argument("--overwrite", action="store_true", help="replace a frozen version")
    parser.add_argument("--list", action="store_true", help="show the sets without writing")
    parser.add_argument(
        "--store-optima",
        action="store_true",
        help="prove and store the optimum of each instance in an existing set",
    )
    parser.add_argument("--seconds", type=float, default=120.0, help="exact-search budget each")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks")
    args = parser.parse_args()
    if args.list or args.set is None:
        print("available sets:", ", ".join(sorted(BUILDERS)))
        return
    if args.store_optima:
        store_optima(args.set, args.out, args.seconds, args.overwrite)
        return
    bench = BUILDERS[args.set]()
    reference = {
        inst.instance_id: InstanceReference(
            cost=Rational.parse(inst.metadata["known_upper_bound"]),
            status="feasible",
            producer="naive",
            detail={"witness_actions": inst.metadata["witness_actions"]},
        )
        for inst in bench.instances
        if "known_upper_bound" in inst.metadata and inst.instance_id not in bench.reference
    }
    path = bench.with_reference(reference).write(args.out, overwrite=args.overwrite)
    print(summarise(bench))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
