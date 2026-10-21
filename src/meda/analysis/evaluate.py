"""`evaluate(producer, benchmark, cfg) -> DataFrame` (SPEC 12.2).

One harness grades every producer the same way. A **producer** turns an instance into a
`SolveResult`: solvers do so directly, agents through `run_episode`. For each instance of a
frozen set, `evaluate`:

1. runs the producer;
2. validates its DAG — complete if the plan succeeded, partial otherwise. A DAG that fails
   validation raises: it is a bug in the producer or the core, never a bad score;
3. computes the DAG metrics (`dag_metrics`);
4. computes gaps against the set's answer key, exactly (from rational costs), and only for a
   plan that met all demand: an unfinished plan's cost is not comparable.

Gap columns (each `(cost - ref) / ref`):
- `gap_to_optimal` where the reference is a proved optimum, and only there;
- `gap_to_best_known` against the stored best known cost (equal to `gap_to_optimal` where that
  is proved). Negative means the producer beat every stored plan;
- `gap_to_lower_bound` where a proved lower bound is stored: an upper bound on the true gap to
  optimal. `bound_ratio` = best known / lower bound says how loose that interval is.
Gaps against baselines are added by `compare.add_baseline_gaps`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol

import pandas as pd

from meda.agents.base import Agent, run_episode
from meda.analysis.dag_metrics import DagMetrics, ScheduleConfig, dag_metrics
from meda.core.composition import Rational
from meda.core.costs import CostWeights
from meda.core.dag import validate
from meda.core.problem import Instance
from meda.envs.sample_prep_env import EnvConfig, FixedInstances, InstanceSource, SamplePrepEnv
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.base import SolveBudget, Solver, SolveResult
from meda.solvers.fallback import ConstructiveCostToFinish


class Producer(Protocol):
    @property
    def name(self) -> str: ...

    def produce(self, instance: Instance) -> SolveResult: ...


@dataclass(slots=True)
class SolverProducer:
    solver: Solver
    budget: SolveBudget

    @property
    def name(self) -> str:
        return self.solver.name

    def produce(self, instance: Instance) -> SolveResult:
        return self.solver.solve(instance, self.budget)


@dataclass(slots=True)
class AgentProducer:
    """Greedy (`deterministic`, k = 1), or the cheapest successful plan of k sampled episodes."""

    agent: Agent
    env: SamplePrepEnv
    deterministic: bool = True
    best_of_k: int = 1

    def __post_init__(self) -> None:
        if self.best_of_k < 1:
            raise ValueError(f"best_of_k must be >= 1, got {self.best_of_k}")
        if self.deterministic and self.best_of_k > 1:
            raise ValueError("best_of_k > 1 needs deterministic=false: greedy repeats one plan")

    @property
    def name(self) -> str:
        return self.agent.name

    def produce(self, instance: Instance) -> SolveResult:
        start = time.perf_counter()
        best: SolveResult | None = None
        for _ in range(self.best_of_k):
            r = run_episode(self.agent, self.env, instance, self.deterministic)
            if best is None or _better(r, best):
                best = r
        assert best is not None
        return SolveResult(
            best.instance_id,
            best.producer,
            best.status,
            best.dag,
            best.totals,
            best.cost,
            time.perf_counter() - start,
        )


def _better(a: SolveResult, b: SolveResult) -> bool:
    """Success beats failure; among equals, the lower cost."""
    a_ok, b_ok = a.status != "failed", b.status != "failed"
    if a_ok != b_ok:
        return a_ok
    return a.cost is not None and (b.cost is None or a.cost < b.cost)


def build_env(
    cfg: Mapping[str, Any], instances: list[Instance], source: InstanceSource | None = None
) -> SamplePrepEnv:
    """The env agents are evaluated (and trained) in, from the `env` and `costs` configs. It
    draws from `instances`, or from `source` (e.g. a generator) when one is given."""
    weights = CostWeights.from_dict(cfg["costs"])
    return SamplePrepEnv(
        EnvConfig.from_dict(cfg["env"]),
        FixedInstances(instances) if source is None else source,
        weights,
        ConstructiveCostToFinish(weights),
    )


def ratio(a: Rational, b: Rational, instance_id: str) -> Rational:
    """a / b, exact. `b` must be > 0."""
    if b <= 0:
        raise ValueError(f"{instance_id}: reference cost {b} must be > 0 to take a ratio")
    return Rational.of(a.num * b.den, a.den * b.num)


def relative_gap(cost: Rational, reference: Rational, instance_id: str) -> float:
    """(cost - reference) / reference, exact until the final float."""
    return float(ratio(cost - reference, reference, instance_id))


SUCCESS_STATUSES = ("optimal", "feasible")
METRIC_COLUMNS = tuple(f.name for f in fields(DagMetrics))


def evaluate(producer: Producer, benchmark: BenchmarkSet, schedule: ScheduleConfig) -> pd.DataFrame:
    """One row per instance of `benchmark` (see module docstring for the columns)."""
    rows: list[dict[str, Any]] = []
    for instance in benchmark.instances:
        result = producer.produce(instance)
        if result.instance_id != instance.instance_id:
            raise ValueError(
                f"{producer.name} answered {result.instance_id} for {instance.instance_id}"
            )
        success = result.status in SUCCESS_STATUSES
        metrics: DagMetrics | None = None
        if result.dag is not None:
            validate(result.dag, instance, complete=success).raise_if_invalid()
            metrics = dag_metrics(result.dag, instance, schedule)
        elif success:
            raise ValueError(
                f"{producer.name} reports {result.status} on {instance.instance_id} without a DAG"
            )

        ref = benchmark.reference.get(instance.instance_id)
        row: dict[str, Any] = {
            "benchmark": benchmark.full_name,
            "instance_id": instance.instance_id,
            "producer": producer.name,
            "status": result.status,
            "success": success,
            "cost": float(result.cost) if result.cost is not None else None,
            "cost_exact": str(result.cost) if result.cost is not None else None,
            "reference_cost": str(ref.cost) if ref else None,
            "reference_status": ref.status if ref else None,
            "lower_bound": str(ref.lower_bound) if ref and ref.lower_bound is not None else None,
            "gap_to_optimal": None,
            "gap_to_best_known": None,
            "gap_to_lower_bound": None,
            "bound_ratio": None,
            "num_targets": len(instance.targets),
            "total_demand": sum(t.demand for t in instance.targets),
            # The input axes of SPEC 1.3, for gap-vs-axis curves (12.3).
            "num_fluids": instance.num_fluids,
            "max_mix_volume": instance.chip.max_mix_volume,
            "storage_capacity": instance.chip.storage_capacity,
            "max_droplet_volume": max(t.droplet_volume for t in instance.targets),
            "wall_time": result.wall_time,
            "nodes_expanded": result.nodes_expanded,
        }
        if ref is not None and ref.lower_bound is not None and ref.lower_bound > 0:
            row["bound_ratio"] = float(ratio(ref.cost, ref.lower_bound, instance.instance_id))
        if success and ref is not None:
            assert result.cost is not None
            iid = instance.instance_id
            row["gap_to_best_known"] = relative_gap(result.cost, ref.cost, iid)
            if ref.status == "optimal":
                row["gap_to_optimal"] = row["gap_to_best_known"]
            if ref.lower_bound is not None and ref.lower_bound > 0:  # a 0 bound says nothing
                row["gap_to_lower_bound"] = relative_gap(result.cost, ref.lower_bound, iid)
        row.update(metrics.as_dict() if metrics else dict.fromkeys(METRIC_COLUMNS))
        rows.append(row)
    return pd.DataFrame(rows)
