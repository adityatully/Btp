"""Hydra entry point for training (SPEC section 13).

    uv run python -m meda.train agent=tabular_q
    uv run python -m meda.train agent=tabular_q train.benchmark=tiny_exact_v1 agent.episodes=40000

Tabular Q (M6) trains on a frozen benchmark set: it keys on exact states, so it only learns
anything about instances it will meet again, and the M6 acceptance criterion is per instance.
PPO (M8) will train from the sampler instead.

Each run writes into Hydra's run directory: the resolved config, one checkpoint per table, and
the greedy policy graded by the M7 harness (`analysis.evaluate`): `results.parquet`, with the
training statistics as extra columns, and `summary.md`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from meda.agents.tabular_q import TabularQ, TabularQConfig
from meda.analysis.compare import write_summary_md
from meda.analysis.dag_metrics import ScheduleConfig
from meda.analysis.evaluate import AgentProducer, build_env, evaluate
from meda.core.chip import ChipLimits
from meda.core.costs import CostWeights
from meda.generators.benchmark_io import BenchmarkSet
from meda.utils.seeding import seed_everything


def train_tabular_q(cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    """One Q table per instance (the M6 acceptance criterion), then a greedy evaluation."""
    train_cfg = cfg["train"]
    agent_cfg = TabularQConfig.from_dict({**cfg["agent"], "seed": cfg["seed"]})
    weights = CostWeights.from_dict(cfg["costs"])
    bench = BenchmarkSet.read(str(train_cfg["benchmark"]))
    wanted = train_cfg.get("instances")
    instances = [bench.by_id(i) for i in wanted] if wanted else list(bench.instances)
    groups = [[i] for i in instances] if train_cfg.get("per_instance", True) else [instances]

    schedule = ScheduleConfig.from_dict(cfg["eval"]["schedule"])
    tables: list[pd.DataFrame] = []
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for group in groups:
        label = group[0].instance_id if len(group) == 1 else "all"
        print(f"\ntraining on {label} ({len(group)} instance(s), {agent_cfg.episodes} episodes)")
        env = build_env(cfg, group)
        agent = TabularQ(agent_cfg, weights)
        started = time.perf_counter()
        stats = agent.train(env, group, log_every=int(train_cfg.get("log_every", 0)))
        elapsed = time.perf_counter() - started
        agent.save(checkpoints / f"{label}.pkl")

        part = BenchmarkSet(
            bench.name,
            bench.version,
            tuple(group),
            reference={i.instance_id: bench.reference[i.instance_id] for i in group},
        )
        df = evaluate(AgentProducer(agent, env), part, schedule).assign(
            trained_on=label,
            episodes=stats.episodes,
            train_success_rate=stats.success_rate,
            best_cost_seen=stats.best_cost,
            table_entries=stats.table_size,
            train_seconds=elapsed,
        )
        for row in df.itertuples():
            mark = "optimal" if row.gap_to_optimal == 0 else "above optimum"
            print(
                f"  {row.instance_id:24s} greedy={row.cost_exact} optimum={row.reference_cost} "
                f"({mark}); success_rate={stats.success_rate:.1%}, |Q|={stats.table_size}"
            )
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    seed_everything(int(cfg.seed))
    chip = ChipLimits.from_dict(data["chip"])
    print(f"seed={cfg.seed} agent={cfg.agent.name} chip={chip.name} -> {out_dir}")

    name = str(cfg.agent.name)
    if name != "tabular_q":
        raise NotImplementedError(
            f"agent '{name}' is not trainable yet: tabular Q is M6, PPO is M8 (SPEC 16)"
        )
    results = train_tabular_q(data, out_dir)  # type: ignore[arg-type]

    results.to_parquet(out_dir / "results.parquet", index=False)
    write_summary_md(out_dir / "summary.md", results)
    reached = int((results["gap_to_optimal"] == 0).sum())
    print(
        f"\n{reached}/{len(results)} instances reached the stored optimum; "
        f"results in {out_dir / 'results.parquet'} and summary.md"
    )


if __name__ == "__main__":
    main()
