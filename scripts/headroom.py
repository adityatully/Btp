"""Headroom and grading-strength probes (Track 2 experiments E2 and E3; no training).

E2 (`mode=headroom`): small problems in every cell of the constraint grid (dilution / mixture x
one target / several x demand 1 / several x storage loose / tight, plus dilution gradients).
Each is solved by naive, the split tree and exact search; the record says how much cheaper the
proved optimum is than the constructions, and what the optimal plan does differently (sharing,
partial takes, mixes, fluid per kind).

E3 (`mode=grading`): medium problems in the multi-target, multi-demand cells, where exact search
does not finish. Exact search runs at growing time budgets; the record holds the interval
[lower bound, best known] at each, i.e. how much a frontier benchmark could say.

Usage:
    uv run python scripts/headroom.py mode=headroom per_cell=30 seconds=30 out=notes/e2.jsonl
    uv run python scripts/headroom.py mode=grading per_cell=8 budgets=10,60,300 out=notes/e3.jsonl

Arguments are key=value. All draws use the generator (8.1) with the 64/8/1 cost profile and
the depth step-limit rule; seeds are (cell index, draw index), so a rerun is identical.
"""

from __future__ import annotations

import json
import math
import sys
import time
import zlib
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from meda.analysis.dag_metrics import ScheduleConfig, dag_metrics
from meda.core.actions import Mix
from meda.core.chip import ChipLimits
from meda.core.composition import Composition, Rational
from meda.core.costs import CostWeights
from meda.core.dag import MixingDAG
from meda.core.problem import Instance
from meda.core.rules import apply, replay
from meda.core.state import PrepState
from meda.generators.sampler import (
    GeneratorConfig,
    UnwitnessedError,
    is_smooth,
    sample_instance,
    step_limit_base,
)
from meda.solvers.base import SolveBudget
from meda.solvers.baselines.naive import naive_plan
from meda.solvers.baselines.tree import tree_plan
from meda.solvers.exact_search import ExactSearch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = CostWeights.from_dict(yaml.safe_load((ROOT / "configs/costs/geometric.yaml").read_text()))
CHIP = ChipLimits.from_dict(yaml.safe_load((ROOT / "configs/chip/meda_m8.yaml").read_text()))
SCHED = ScheduleConfig(1, 1, None)


def cell_config(name: str, kind: str, targets: str, demand: str, storage: str, size: str) -> dict:
    """A generator config for one cell. `size` small (E2) or medium (E3)."""
    small = size == "small"
    fluids = (
        {"samples": 1, "reagents": 0, "buffers": 1}
        if kind == "dil"
        else {"samples": 1, "reagents": [1, 2] if not small else 1, "buffers": 1}
    )
    tpe = 1 if targets == "one" else ([2, 3] if small else [4, 6])
    return {
        "name": name,
        "num_fluids": fluids,
        "num_experiments": 1,
        "targets_per_experiment": tpe,
        "composition_sampler": "dirichlet_snapped",
        "accuracy_denominator": [4, 16] if small else [8, 64],
        "support_size": 2 if kind == "dil" else [2, 3],
        "demand": 1 if demand == "d1" else ([2, 3] if small else [2, 4]),
        "droplet_volume": [1, "M"],
        "tolerance": 0,
        "max_mix_volume": [3, 8],
        "storage_capacity": [2, 3] if storage == "tight" else 10,
    }


def cells(size: str) -> list[tuple[str, dict, bool]]:
    """(cell name, generator config dict, gradient?) for every cell of the grid."""
    out = []
    for kind in ("dil", "mix"):
        for targets in ("one", "many"):
            for demand in ("d1", "dN"):
                for storage in ("loose", "tight"):
                    if size == "medium" and (targets == "one" or demand == "d1"):
                        continue
                    name = f"{kind}-{targets}-{demand}-{storage}"
                    out.append(
                        (name, cell_config(name, kind, targets, demand, storage, size), False)
                    )
    for demand in ("d1", "dN"):
        for storage in ("loose", "tight"):
            if size == "medium" and demand == "d1":
                continue
            name = f"dilgrad-many-{demand}-{storage}"
            cfg = cell_config(name, "dil", "many", demand, storage, size)
            out.append((name, cfg, True))
    return out


def as_gradient(inst: Instance, rng: np.random.Generator) -> Instance:
    """Replace the targets with a linear dilution gradient: one denominator D, sample shares
    evenly spaced (the structured multi-target case where sharing should pay)."""
    m = inst.chip.max_mix_volume
    dens = [d for d in range(4, 17) if is_smooth(d, m)]
    t = len(inst.targets)
    dens = [d for d in dens if d - 1 >= t] or [max(dens)]
    den = int(dens[int(rng.integers(len(dens)))])
    step = max(1, (den - 1) // (t + 1))
    lo = int(rng.integers(1, den - (t - 1) * step))
    targets = tuple(
        replace(old, composition=Composition.from_ints((lo + i * step, den - lo - i * step), den))
        for i, old in enumerate(inst.targets)
    )
    probe = replace(inst, targets=targets, step_limit=10_000)
    witness = naive_plan(probe) or tree_plan(probe)
    if witness is None:
        raise UnwitnessedError(f"gradient '{inst.instance_id}': no witness")
    limit = max(math.ceil(4 * step_limit_base(probe)), len(witness))
    return replace(probe, step_limit=limit)


def plan_stats(inst: Instance, actions: tuple | None) -> dict[str, Any] | None:
    if actions is None:
        return None
    state = PrepState.initial(inst)
    partial = 0
    for act in actions:
        if isinstance(act, Mix):
            for src, take in ((act.a, act.k), (act.b, act.l)):
                if not src.is_reservoir and take < state.droplet(src.index).volume:
                    partial += 1
        state, _, _ = apply(state, act, inst)
    _, totals, deltas = replay(inst, actions)
    dag = MixingDAG.from_deltas(inst, deltas, totals, "probe")
    met = dag_metrics(dag, inst, SCHED).as_dict()
    keep = (
        "mix_ops",
        "sample_volume",
        "reagent_volume",
        "buffer_volume",
        "waste_volume",
        "sharing_count",
        "distinct_intermediates",
        "peak_droplets",
    )
    out = {k: met[k] for k in keep}
    out["partial_takes"] = partial
    out["steps"] = len(actions)
    out["cost"] = _f(WEIGHTS.scalar(totals, inst.fluids))
    return out


def _f(r: Rational | None) -> float | None:
    return None if r is None else r.num / r.den


def describe(inst: Instance) -> dict[str, Any]:
    return {
        "M": inst.chip.max_mix_volume,
        "storage": inst.chip.storage_capacity,
        "F": len(inst.fluids),
        "T": len(inst.targets),
        "deliveries": sum(t.demand for t in inst.targets),
        "dens": [t.composition.den for t in inst.targets],
        "support": [sum(1 for x in t.composition.num if x) for t in inst.targets],
        "volumes": [t.droplet_volume for t in inst.targets],
        "demands": [t.demand for t in inst.targets],
        "step_limit": inst.step_limit,
    }


def run_one(task: tuple) -> dict[str, Any]:
    mode, cell, cfg_dict, gradient, index, seconds, budgets = task
    rng = np.random.default_rng([zlib.crc32(cell.encode()), index])
    rec: dict[str, Any] = {"mode": mode, "cell": cell, "index": index}
    try:
        inst = sample_instance(GeneratorConfig.from_dict(cfg_dict), CHIP, rng, WEIGHTS)
        if gradient:
            inst = as_gradient(inst, rng)
    except (UnwitnessedError, ValueError) as e:
        rec["skipped"] = str(e)[:200]
        return rec
    rec.update(describe(inst))
    rec["naive"] = plan_stats(inst, naive_plan(inst))
    rec["tree"] = plan_stats(inst, tree_plan(inst))
    search = ExactSearch(WEIGHTS)
    if mode == "headroom":
        t0 = time.perf_counter()
        plan, cost, stats = search.search(inst, SolveBudget(max_nodes=400_000, max_seconds=seconds))
        rec["exact"] = {
            "status": "timeout" if stats.stopped else ("optimal" if plan else "infeasible"),
            "cost": _f(cost),
            "lower_bound": _f(stats.lower_bound),
            "nodes": stats.nodes_expanded,
            "seconds": round(time.perf_counter() - t0, 2),
            "plan": plan_stats(inst, plan),
        }
    else:
        rec["exact"] = []
        for secs in budgets:
            t0 = time.perf_counter()
            plan, cost, stats = search.search(inst, SolveBudget(max_seconds=secs))
            rec["exact"].append(
                {
                    "budget_seconds": secs,
                    "status": "timeout" if stats.stopped else ("optimal" if plan else "infeasible"),
                    "cost": _f(cost),
                    "lower_bound": _f(stats.lower_bound),
                    "nodes": stats.nodes_expanded,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
            if not stats.stopped:
                break
    return rec


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    mode = args.get("mode", "headroom")
    per_cell = int(args.get("per_cell", 30))
    seconds = float(args.get("seconds", 30))
    budgets = [float(x) for x in args.get("budgets", "10,60,300").split(",")]
    workers = int(args.get("workers", 7))
    out = Path(args.get("out", f"notes/{mode}.jsonl"))
    size = "small" if mode == "headroom" else "medium"
    tasks = [
        (mode, name, cfg, grad, i, seconds, budgets)
        for name, cfg, grad in cells(size)
        for i in range(per_cell)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out.open("w") as fh, Pool(workers) as pool:
        for rec in pool.imap_unordered(run_one, tasks):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(tasks)}", flush=True)
    print(f"wrote {done} records to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
