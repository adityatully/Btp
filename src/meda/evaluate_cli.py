"""Hydra entry point for evaluation (SPEC 12.2).

    uv run python -m meda.evaluate_cli eval.producers=[random,naive,exact_search] \
        eval.benchmark=tiny_exact_v1

Grades each producer (and each baseline) on one frozen set and writes into Hydra's run
directory: `results.parquet` (one row per producer and instance), `summary.md` and `plots/`.
Tabular Q is graded by `meda.train`, which owns its per-instance tables; PPO joins in M8.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from meda.agents.random_agent import RandomAgent
from meda.analysis.compare import add_baseline_gaps, write_summary_md
from meda.analysis.dag_metrics import ScheduleConfig
from meda.analysis.evaluate import AgentProducer, Producer, SolverProducer, build_env, evaluate
from meda.analysis.plots import write_plots
from meda.core.costs import CostWeights
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.base import SolveBudget, Solver
from meda.solvers.baselines.bitscan import BitScanSolver
from meda.solvers.baselines.naive import NaiveSolver
from meda.solvers.baselines.tree import TreeSolver
from meda.solvers.exact_search import ExactSearch
from meda.utils.seeding import seed_everything

SOLVERS: dict[str, Callable[[CostWeights], Solver]] = {
    "naive": NaiveSolver,
    "bitscan": BitScanSolver,
    "tree": TreeSolver,
    "exact_search": ExactSearch,
}
PRODUCERS = ("random", *SOLVERS)


def make_producer(name: str, cfg: dict[str, Any], bench: BenchmarkSet) -> Producer:
    ev = cfg["eval"]
    if name == "random":
        env = build_env(cfg, list(bench.instances))
        return AgentProducer(
            RandomAgent(seed=int(cfg["seed"])), env, bool(ev["deterministic"]), int(ev["best_of_k"])
        )
    if name in SOLVERS:
        weights = CostWeights.from_dict(cfg["costs"])
        return SolverProducer(SOLVERS[name](weights), SolveBudget(max_seconds=ev["solver_seconds"]))
    raise ValueError(f"unknown producer {name!r}; choose from {list(PRODUCERS)}")


def run(cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    ev = cfg["eval"]
    bench = BenchmarkSet.read(str(ev["benchmark"]))
    schedule = ScheduleConfig.from_dict(ev["schedule"])
    names = list(dict.fromkeys([*ev["producers"], *ev["baselines"]]))
    by_name: dict[str, pd.DataFrame] = {}
    for name in names:
        print(f"evaluating {name} on {bench.full_name} ({len(bench)} instances)")
        by_name[name] = evaluate(make_producer(name, cfg, bench), bench, schedule)
    baselines = {b: by_name[b] for b in ev["baselines"]}
    results = pd.concat(
        [add_baseline_gaps(df, baselines) for df in by_name.values()], ignore_index=True
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_dir / "results.parquet", index=False)
    write_summary_md(out_dir / "summary.md", results)
    write_plots(results, out_dir / "plots", ev["baselines"][0] if ev["baselines"] else None)
    return results


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    seed_everything(int(cfg.seed))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    results = run(data, out_dir)  # type: ignore[arg-type]
    for name, g in results.groupby("producer", sort=False):
        at_opt = int((g["gap_to_optimal"] == 0).sum())
        print(
            f"  {name:14s} success {g['success'].mean():.0%}  "
            f"at optimum {at_opt}/{int(g['gap_to_optimal'].notna().sum())}"
        )
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
