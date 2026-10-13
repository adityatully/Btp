"""Hydra entry point for training (SPEC section 13).

    uv run python -m meda.train agent=tabular_q
    uv run python -m meda.train agent=tabular_q train.benchmark=tiny_exact_v1 agent.episodes=40000

Tabular Q (M6) trains on a frozen benchmark set: it keys on exact states, so it only learns
anything about instances it will meet again, and the M6 acceptance criterion is per instance.
PPO (M8) will train from the sampler instead.

Each run writes into Hydra's run directory: the resolved config, one checkpoint per table, and
`results.json` with the cost the greedy policy reaches against the stored optimum.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from meda.agents.base import run_episode
from meda.agents.tabular_q import TabularQ, TabularQConfig
from meda.core.chip import ChipLimits
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.envs.sample_prep_env import EnvConfig, FixedInstances, SamplePrepEnv
from meda.generators.benchmark_io import BenchmarkSet, InstanceReference
from meda.solvers.fallback import NaiveCostToFinish
from meda.utils.seeding import seed_everything


def build_env(cfg: dict[str, Any], instances: list[Instance]) -> SamplePrepEnv:
    weights = CostWeights.from_dict(cfg["costs"])
    return SamplePrepEnv(
        EnvConfig.from_dict(cfg["env"]),
        FixedInstances(instances),
        weights,
        NaiveCostToFinish(weights),
    )


def train_tabular_q(cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    """One Q table per instance (the M6 acceptance criterion), then a greedy evaluation."""
    train_cfg = cfg["train"]
    agent_cfg = TabularQConfig.from_dict({**cfg["agent"], "seed": cfg["seed"]})
    weights = CostWeights.from_dict(cfg["costs"])
    bench = BenchmarkSet.read(str(train_cfg["benchmark"]))
    wanted = train_cfg.get("instances")
    instances = [bench.by_id(i) for i in wanted] if wanted else list(bench.instances)
    groups = [[i] for i in instances] if train_cfg.get("per_instance", True) else [instances]

    rows: list[dict[str, Any]] = []
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

        for instance in group:
            result = run_episode(agent, env, instance, deterministic=True)
            stored: InstanceReference | None = bench.reference.get(instance.instance_id)
            cost = float(result.cost) if result.cost is not None else None
            optimum = float(stored.cost) if stored else None
            gap = None if cost is None or not optimum else (cost - optimum) / abs(optimum)
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "trained_on": label,
                    "status": result.status,
                    "cost": cost,
                    "optimal": optimum,
                    "optimal_status": stored.status if stored else None,
                    "gap_to_optimal": gap,
                    "episodes": stats.episodes,
                    "success_rate": stats.success_rate,
                    "best_cost_seen": stats.best_cost,
                    "table_entries": stats.table_size,
                    "train_seconds": elapsed,
                }
            )
            mark = "optimal" if gap is not None and abs(gap) < 1e-12 else "above optimum"
            print(
                f"  {instance.instance_id:24s} greedy={cost} optimum={optimum} "
                f"({mark}); success_rate={stats.success_rate:.1%}, |Q|={stats.table_size}"
            )
    return rows


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
    rows = train_tabular_q(data, out_dir)  # type: ignore[arg-type]

    (out_dir / "results.json").write_text(json.dumps(rows, indent=1))
    reached = [
        r for r in rows if r["gap_to_optimal"] is not None and abs(r["gap_to_optimal"]) < 1e-12
    ]
    print(
        f"\n{len(reached)}/{len(rows)} instances reached the stored optimum; "
        f"results in {out_dir / 'results.json'}"
    )


if __name__ == "__main__":
    main()
