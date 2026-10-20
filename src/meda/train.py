"""Hydra entry point for training (SPEC section 13).

    uv run python -m meda.train agent=tabular_q
    uv run python -m meda.train agent=tabular_q train.benchmark=tiny_exact_v1 agent.episodes=40000

    uv run python -m meda.train agent=ppo train.instances=[tiny-sharing-m2] env.L_pad=128

Both agents train on instances from a frozen benchmark set (`train.benchmark`,
`train.instances`). Tabular Q (M6) needs that: it keys on exact states, so it only learns about
instances it will meet again, and the M6 acceptance criterion is per instance. PPO (M8) trains
one network on the selected instances, which is acceptance (a); training from a generator
follows the stage design. A PPO run writes `metrics.jsonl`, TensorBoard logs and
`checkpoints/last.pt` (resume with `train.resume=<path>`).

Each run writes into Hydra's run directory: the resolved config, one checkpoint per table, and
the greedy policy graded by the M7 harness (`analysis.evaluate`): `results.parquet`, with the
training statistics as extra columns, and `summary.md`.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from meda.agents.ppo.ppo import PPOConfig, PPOTrainer, VecEnv
from meda.agents.tabular_q import TabularQ, TabularQConfig
from meda.analysis.compare import write_summary_md
from meda.analysis.dag_metrics import ScheduleConfig
from meda.analysis.evaluate import AgentProducer, build_env, evaluate
from meda.core.chip import ChipLimits
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.envs.sample_prep_env import EnvConfig
from meda.generators.benchmark_io import BenchmarkSet
from meda.utils.checkpoint import load_checkpoint, save_checkpoint
from meda.utils.logging import MetricsLogger
from meda.utils.seeding import seed_everything


def training_instances(train_cfg: dict[str, Any]) -> tuple[BenchmarkSet, list[Instance]]:
    """The frozen set named by `train.benchmark` and the instances `train.instances` picks."""
    bench = BenchmarkSet.read(str(train_cfg["benchmark"]))
    wanted = train_cfg.get("instances")
    return bench, [bench.by_id(i) for i in wanted] if wanted else list(bench.instances)


def subset(bench: BenchmarkSet, instances: list[Instance]) -> BenchmarkSet:
    """`bench` restricted to `instances`, keeping their answer keys."""
    ids = [i.instance_id for i in instances]
    return BenchmarkSet(
        bench.name,
        bench.version,
        tuple(instances),
        reference={i: bench.reference[i] for i in ids if i in bench.reference},
    )


def train_tabular_q(cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    """One Q table per instance (the M6 acceptance criterion), then a greedy evaluation."""
    train_cfg = cfg["train"]
    agent_cfg = TabularQConfig.from_dict({**cfg["agent"], "seed": cfg["seed"]})
    weights = CostWeights.from_dict(cfg["costs"])
    bench, instances = training_instances(train_cfg)
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

        df = evaluate(AgentProducer(agent, env), subset(bench, group), schedule).assign(
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


def resolve_device(name: str) -> torch.device:
    """`auto` = cuda if available, else cpu (mps is opt-in: small batches run faster on cpu)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_vector_env(
    cfg: dict[str, Any], instances: list[Instance], num_envs: int, mode: str
) -> VecEnv:
    """`num_envs` copies of the training env, autoresetting in the same step (SPEC 9.1)."""
    factory = functools.partial(build_env, cfg, instances)
    if mode == "sync":
        return SyncVectorEnv([factory] * num_envs, autoreset_mode=AutoresetMode.SAME_STEP)
    if mode == "async":
        return AsyncVectorEnv([factory] * num_envs, autoreset_mode=AutoresetMode.SAME_STEP)
    raise ValueError(f"train.vector must be 'sync' or 'async', got {mode!r}")


def train_ppo(cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    """Train one PPO network on the selected frozen instances, then grade its greedy policy."""
    train_cfg = cfg["train"]
    ppo_cfg = PPOConfig.from_dict(cfg["agent"])
    bench, instances = training_instances(train_cfg)
    graded = subset(bench, instances)
    schedule = ScheduleConfig.from_dict(cfg["eval"]["schedule"])
    sizes = EnvConfig.from_dict(cfg["env"]).sizes
    device = resolve_device(str(train_cfg["device"]))
    checkpoint = out_dir / "checkpoints" / "last.pt"
    envs = make_vector_env(cfg, instances, ppo_cfg.num_envs, str(train_cfg["vector"]))
    logger = MetricsLogger(out_dir, tensorboard=True)
    try:
        trainer = PPOTrainer(ppo_cfg, sizes, envs, device, seed=int(cfg["seed"]))
        if train_cfg.get("resume"):
            load_checkpoint(train_cfg["resume"], trainer)
            print(f"resumed from {train_cfg['resume']} at update {trainer.update}")
        else:
            trainer.start()
        eval_env = build_env(cfg, instances)
        every_ckpt = int(train_cfg["checkpoint_every_updates"])
        every_eval = int(train_cfg["eval_every_updates"])
        print(f"PPO on {len(instances)} instance(s), {ppo_cfg.num_updates} updates, {device}")
        while not trainer.done:
            metrics = trainer.iteration()
            logger.log(trainer.update, metrics)
            logger.console(trainer.update, ppo_cfg.num_updates, metrics)
            if every_ckpt and trainer.update % every_ckpt == 0:
                save_checkpoint(checkpoint, trainer)
            if every_eval and trainer.update % every_eval == 0:
                df = evaluate(AgentProducer(trainer.agent, eval_env), graded, schedule)
                logger.log(trainer.update, _eval_summary(df), prefix="eval")
        save_checkpoint(checkpoint, trainer)
        trainer.agent.save(out_dir / "checkpoints" / "agent.pt")
        df = evaluate(AgentProducer(trainer.agent, eval_env), graded, schedule)
        return df.assign(updates=trainer.update, env_steps=trainer.global_step)
    finally:
        envs.close()
        logger.close()


def _eval_summary(df: pd.DataFrame) -> dict[str, float]:
    """Greedy success rate and mean gaps over successful plans."""
    out = {"success_rate": float(df["success"].mean())}
    for col in ("gap_to_best_known", "gap_to_optimal"):
        values = df[col].dropna()
        out[col] = float(values.mean()) if len(values) else float("nan")
    return out


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    seed_everything(int(cfg.seed))
    chip = ChipLimits.from_dict(data["chip"])
    print(f"seed={cfg.seed} agent={cfg.agent.name} chip={chip.name} -> {out_dir}")

    name = str(cfg.agent.name)
    if name == "tabular_q":
        results = train_tabular_q(data, out_dir)  # type: ignore[arg-type]
    elif name == "ppo":
        results = train_ppo(data, out_dir)  # type: ignore[arg-type]
    else:
        raise NotImplementedError(f"agent '{name}' is not trainable (tabular_q, ppo)")

    results.to_parquet(out_dir / "results.parquet", index=False)
    write_summary_md(out_dir / "summary.md", results)
    reached = int((results["gap_to_optimal"] == 0).sum())
    print(
        f"\n{reached}/{len(results)} instances reached the stored optimum; "
        f"results in {out_dir / 'results.parquet'} and summary.md"
    )


if __name__ == "__main__":
    main()
