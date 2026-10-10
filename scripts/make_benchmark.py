"""Build and freeze a versioned benchmark set (SPEC 8.4).

    uv run python scripts/make_benchmark.py tiny_exact_v1
    uv run python scripts/make_benchmark.py val_stage1_v1
    uv run python scripts/make_benchmark.py --list

A published version is frozen: the script refuses to overwrite one unless `--overwrite` is
given, and the right move after a definition changes is to bump the version. Chip and cost
configs are read straight from `configs/`, because a set may mix chips (the hand-checked cases
of SPEC 17.3 use M = 2, M = 4 and M = 8).

Optima are **not** stored here: `tiny_exact_v1` gets its answer key from the exact search in M5
(`BenchmarkSet.with_reference`). This script stores the naive witness cost, a feasible upper
bound, for every instance it writes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from meda.core.chip import ChipLimits
from meda.core.composition import ZERO, Composition, Rational
from meda.core.costs import CostWeights
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet, InstanceReference, instance_hash
from meda.generators.sampler import GeneratorConfig, dilution_grid, sample_instances
from meda.solvers.baselines.naive import naive_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SB = (Fluid(0, "S", FluidKind.SAMPLE), Fluid(1, "B", FluidKind.BUFFER))


def load_chip(name: str) -> ChipLimits:
    data = OmegaConf.to_container(OmegaConf.load(CONFIGS / "chip" / f"{name}.yaml"))
    assert isinstance(data, dict)
    return ChipLimits.from_dict(data)  # type: ignore[arg-type]


def load_weights() -> CostWeights:
    data = OmegaConf.to_container(OmegaConf.load(CONFIGS / "costs" / "default.yaml"))
    assert isinstance(data, dict)
    return CostWeights.from_dict(data)  # type: ignore[arg-type]


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


def dilution(
    instance_id: str,
    shares: list[tuple[int, int]],
    chip: ChipLimits,
    demand: int = 1,
    droplet_volume: int = 1,
    note: str = "",
) -> Instance:
    """A sample+buffer instance whose step limit is twice its naive witness plan."""
    targets = tuple(
        Target(i, 0, Composition.from_ints((n, d - n), d), ZERO, droplet_volume, demand)
        for i, (n, d) in enumerate(shares)
    )
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
            "contents": "SPEC 17.3 hand-checked cases + every distinct dilution n/8 at M = 2",
        },
        description=(
            "Instances small enough for exact search. Optima are stored by the exact search in "
            "M5; until then each instance carries the naive witness cost as an upper bound."
        ),
    )


def build_val_stage1() -> BenchmarkSet:
    """In-distribution validation for curriculum stage 1 (`tiny_dilution`).

    Stage 1 has only 15 distinct problems (the reduced fractions n/16), so this set is the part
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
            "tiny_exact_v1. Stage 1 holds only 15 distinct problems, so training must exclude "
            "these hashes."
        ),
    )


BUILDERS: dict[str, Callable[[], BenchmarkSet]] = {
    "tiny_exact_v1": build_tiny_exact,
    "val_stage1_v1": build_val_stage1,
}


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
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks")
    args = parser.parse_args()
    if args.list or args.set is None:
        print("available sets:", ", ".join(sorted(BUILDERS)))
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
        if "known_upper_bound" in inst.metadata
    }
    path = bench.with_reference(reference).write(args.out, overwrite=args.overwrite)
    print(summarise(bench))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
