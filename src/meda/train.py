"""Hydra entry point for training (SPEC section 13).

    uv run python -m meda.train agent=tabular_q
    uv run python -m meda.train agent=tabular_q train.benchmark=tiny_exact_v1 agent.episodes=40000

    uv run python -m meda.train agent=ppo train.instances=[tiny-sharing-m2] env.L_pad=128

    uv run python -m meda.train agent=ppo train.source=generator generator=dilution_multi \
        train.benchmark=val_stage1_v2

    uv run python -m meda.train agent=ppo train.source=regime regime=expanding costs=geometric

By default both agents train on instances from a frozen benchmark set (`train.benchmark`,
`train.instances`). Tabular Q (M6) needs that: it keys on exact states, so it only learns about
instances it will meet again, and the M6 acceptance criterion is per instance. PPO (M8) trains
one network on the selected instances, which is acceptance (a). With `train.source=generator`,
PPO instead draws a fresh problem from the `generator` config every episode, never one held in
`train.exclude` or in `train.benchmark`, which then serves only as the evaluation set. With
`train.source=regime`, PPO draws from the cells of the stage design (`stages`) with the shares
the training regime (`regime`) sets, validates on every stage's set each
`train.eval_every_updates` and lets the regime promote (SPEC 8.3). A PPO run writes
`metrics.jsonl`, TensorBoard logs and `checkpoints/last.pt` (resume with `train.resume=<path>`).

Each run writes into Hydra's run directory: the resolved config, one checkpoint per table, and
the greedy policy graded by the M7 harness (`analysis.evaluate`): `results.parquet`, with the
training statistics as extra columns, and `summary.md`.
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from meda.agents.ppo.collector import ActorPool
from meda.agents.ppo.ppo import PPOConfig, PPOTrainer, VecEnv
from meda.agents.tabular_q import TabularQ, TabularQConfig
from meda.analysis.compare import write_summary_md
from meda.analysis.dag_metrics import ScheduleConfig
from meda.analysis.evaluate import AgentProducer, build_env, evaluate
from meda.core.chip import ChipLimits
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.envs.sample_prep_env import EnvConfig, InstanceSource
from meda.generators.benchmark_io import BenchmarkSet
from meda.generators.regime import Regime, RegimeConfig
from meda.generators.sampler import GeneratorConfig, GeneratorSource
from meda.generators.stages import StageDesign
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
    if str(train_cfg.get("source", "benchmark")) != "benchmark":
        raise ValueError("tabular Q trains on frozen instances only (train.source=benchmark)")
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


def resolve_device(name: str, gpu_memory_fraction: float = 1.0) -> torch.device:
    """`auto` = cuda if available, else cpu (mps is opt-in: small batches run faster on cpu).
    On CUDA, this process may allocate at most `gpu_memory_fraction` of the GPU's memory
    (`train.gpu_memory_fraction`): the workstation is shared, so a run that needs more fails
    with an out-of-memory error instead of taking the GPU from other users."""
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    )
    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError(f"train.gpu_memory_fraction must be in (0, 1], got {gpu_memory_fraction}")
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, index)
    return device


def training_source(cfg: dict[str, Any], bench: BenchmarkSet) -> GeneratorSource | None:
    """The generator to train from (`train.source=generator`), excluding every problem of the
    sets in `train.exclude` and of `bench`; None to train on the benchmark's instances."""
    train_cfg = cfg["train"]
    source = str(train_cfg.get("source", "benchmark"))
    if source == "benchmark":
        return None
    if source != "generator":
        raise ValueError(f"train.source must be 'benchmark' or 'generator', got {source!r}")
    held: set[str] = set(bench.hashes())
    for name in train_cfg.get("exclude") or ():
        held |= BenchmarkSet.read(str(name)).hashes()
    return GeneratorSource(
        GeneratorConfig.from_dict(cfg["generator"]),
        ChipLimits.from_dict(cfg["chip"]),
        CostWeights.from_dict(cfg["costs"]),
        frozenset(held),
    )


def make_training_envs(
    cfg: dict[str, Any],
    instances: list[Instance],
    num_envs: int,
    source: InstanceSource | None = None,
) -> VecEnv | ActorPool:
    """The PPO training envs for `train.collector`: `lockstep` (a vector env, `train.vector`)
    or `workers` (an `ActorPool`: one process per env, each with a policy copy; SPEC 11.4)."""
    train_cfg = cfg["train"]
    collector = str(train_cfg.get("collector", "lockstep"))
    if collector == "lockstep":
        return make_vector_env(cfg, instances, num_envs, str(train_cfg["vector"]), source)
    if collector == "workers":
        factory = functools.partial(build_env, cfg, instances, source)
        return ActorPool([factory] * num_envs, PPOConfig.from_dict(cfg["agent"]).net)
    raise ValueError(f"train.collector must be 'lockstep' or 'workers', got {collector!r}")


def make_vector_env(
    cfg: dict[str, Any],
    instances: list[Instance],
    num_envs: int,
    mode: str,
    source: InstanceSource | None = None,
) -> VecEnv:
    """`num_envs` copies of the training env, autoresetting in the same step (SPEC 9.1).

    `copy=False`: the vector env hands back its own observation buffer instead of a deep copy
    of every padded observation (about 150 MB a step at 32 envs and L_pad 16384, which made
    the main process the bottleneck); the trainer copies what it keeps, trimmed, before the
    next step (`rollout.split_obs`)."""
    factory = functools.partial(build_env, cfg, instances, source)
    reset = AutoresetMode.SAME_STEP
    if mode == "sync":
        return SyncVectorEnv([factory] * num_envs, autoreset_mode=reset, copy=False)
    if mode == "async":
        return AsyncVectorEnv([factory] * num_envs, autoreset_mode=reset, copy=False)
    raise ValueError(f"train.vector must be 'sync' or 'async', got {mode!r}")


def train_ppo(cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    """Train one PPO network on the selected frozen instances or on a generator, then grade
    its greedy policy on the selected frozen instances."""
    train_cfg = cfg["train"]
    ppo_cfg = PPOConfig.from_dict(cfg["agent"])
    bench, instances = training_instances(train_cfg)
    source = training_source(cfg, bench)
    graded = subset(bench, instances)
    schedule = ScheduleConfig.from_dict(cfg["eval"]["schedule"])
    sizes = EnvConfig.from_dict(cfg["env"]).sizes
    device = resolve_device(
        str(train_cfg["device"]), float(train_cfg.get("gpu_memory_fraction", 1.0))
    )
    checkpoint = out_dir / "checkpoints" / "last.pt"
    envs = make_training_envs(cfg, instances, ppo_cfg.num_envs, source)
    logger = MetricsLogger(out_dir, tensorboard=True)
    interrupted = False
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
        drawn = (
            f"{len(instances)} instance(s)"
            if source is None
            else f"generator '{source.config.name}' ({len(source.exclude)} hashes held out)"
        )
        print(f"PPO on {drawn}, {ppo_cfg.num_updates} updates, {device}")
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
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        # Ctrl-C (or `timeout -s INT`) reaches the env workers too: terminate them rather
        # than wait for a step they abandoned (which gymnasium reports as an error).
        envs.close(terminate=interrupted)
        logger.close()


CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def regime_sets(design: StageDesign) -> tuple[dict[str, BenchmarkSet], BenchmarkSet]:
    """Each stage's frozen validation set, and the held-out cells' set."""
    return {s.name: BenchmarkSet.read(s.val) for s in design.stages}, BenchmarkSet.read(
        design.held_out_val
    )


def train_ppo_regime(cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    """Train one PPO network from the stage design's cells under a training regime (SPEC 8.3).

    Every `train.eval_every_updates` the greedy policy is graded on every stage's validation
    set (logged per stage, the forgetting record) and the regime may promote; a new set of
    stage shares is sent to every env worker and takes effect at its next reset. The run ends
    with a grading on every stage's set and on the held-out cells' set."""
    train_cfg = cfg["train"]
    ppo_cfg = PPOConfig.from_dict(cfg["agent"])
    design = StageDesign.from_dict(cfg["stages"])
    regime = Regime(RegimeConfig.from_dict(cfg["regime"]), design, ppo_cfg.total_env_steps)
    weights = CostWeights.from_dict(cfg["costs"])
    graded_costs = OmegaConf.to_container(
        OmegaConf.load(CONFIGS / "costs" / f"{design.costs}.yaml")
    )
    if cfg["costs"] != graded_costs:
        raise ValueError(
            f"stage design '{design.name}' grades under costs/{design.costs}; train with "
            f"costs={design.costs} or the gaps compare plans under different weights"
        )
    # The design names its own chip (platform constants; M and storage are drawn per cell).
    chip_data = OmegaConf.to_container(OmegaConf.load(CONFIGS / "chip" / f"{design.chip}.yaml"))
    chip = ChipLimits.from_dict(chip_data)  # type: ignore[arg-type]
    val_sets, held_out = regime_sets(regime.design)
    held: set[str] = set(held_out.hashes())
    for bench in val_sets.values():
        held |= bench.hashes()
    for name in train_cfg.get("exclude") or ():
        held |= BenchmarkSet.read(str(name)).hashes()
    exclude = frozenset(held)

    def source() -> InstanceSource:
        return regime.design.source(regime.stage_weights(), chip, weights, exclude)

    schedule = ScheduleConfig.from_dict(cfg["eval"]["schedule"])
    sizes = EnvConfig.from_dict(cfg["env"]).sizes
    device = resolve_device(
        str(train_cfg["device"]), float(train_cfg.get("gpu_memory_fraction", 1.0))
    )
    checkpoint = out_dir / "checkpoints" / "last.pt"
    envs = make_training_envs(cfg, [], ppo_cfg.num_envs, source())
    logger = MetricsLogger(out_dir, tensorboard=True)
    interrupted = False
    try:
        trainer = PPOTrainer(ppo_cfg, sizes, envs, device, seed=int(cfg["seed"]))
        if train_cfg.get("resume"):
            extra = load_checkpoint(train_cfg["resume"], trainer)
            regime.load_state_dict(extra["regime"])
            envs.call("set_instance_source", source())
            print(f"resumed from {train_cfg['resume']} at update {trainer.update}")
        else:
            trainer.start()
        every_ckpt = int(train_cfg["checkpoint_every_updates"])
        every_eval = int(train_cfg["eval_every_updates"])
        eval_env = build_env(cfg, [i for b in val_sets.values() for i in b.instances])
        print(
            f"PPO, regime '{regime.cfg.name}' over {regime.names} of design '{design.name}' "
            f"({len(exclude)} hashes held out), {ppo_cfg.num_updates} updates, {device}"
        )
        logger.log(0, regime.stage_weights(), prefix="regime/weight")
        while not trainer.done:
            metrics = trainer.iteration()
            logger.log(trainer.update, metrics)
            logger.console(trainer.update, ppo_cfg.num_updates, metrics)
            # Validate before checkpointing, so a checkpoint holds the regime after the round.
            if every_eval and trainer.update % every_eval == 0:
                started = time.perf_counter()
                results = {}
                for name, bench in val_sets.items():
                    df = evaluate(AgentProducer(trainer.agent, eval_env), bench, schedule)
                    summary = _eval_summary(df)
                    logger.log(trainer.update, summary, prefix=f"eval/{name}")
                    results[name] = (summary["success_rate"], summary["gap_to_best_known"])
                logger.log(trainer.update, {"seconds": time.perf_counter() - started}, "eval")
                if regime.observe(trainer.global_step, results):
                    envs.call("set_instance_source", source())
                    print(f"regime: {regime.events[-1]}; shares now {regime.stage_weights()}")
                    logger.log(trainer.update, regime.stage_weights(), prefix="regime/weight")
            if every_ckpt and trainer.update % every_ckpt == 0:
                save_checkpoint(checkpoint, trainer, {"regime": regime.state_dict()})
        save_checkpoint(checkpoint, trainer, {"regime": regime.state_dict()})
        trainer.agent.save(out_dir / "checkpoints" / "agent.pt")
        (out_dir / "regime.json").write_text(json.dumps(regime.state_dict(), indent=2))
        final_env = build_env(cfg, list(held_out.instances))
        tables = [
            evaluate(AgentProducer(trainer.agent, eval_env), bench, schedule).assign(stage=name)
            for name, bench in val_sets.items()
        ]
        tables.append(
            evaluate(AgentProducer(trainer.agent, final_env), held_out, schedule).assign(
                stage="held_out"
            )
        )
        df = pd.concat(tables, ignore_index=True)
        return df.assign(
            regime=regime.cfg.name, updates=trainer.update, env_steps=trainer.global_step
        )
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        # Ctrl-C (or `timeout -s INT`) reaches the env workers too: terminate them rather
        # than wait for a step they abandoned (which gymnasium reports as an error).
        envs.close(terminate=interrupted)
        logger.close()


def _eval_summary(df: pd.DataFrame) -> dict[str, float]:
    """Greedy success rate and mean gaps over successful plans."""
    out = {"success_rate": float(df["success"].mean())}
    for col in ("gap_to_best_known", "gap_to_optimal"):
        values = df[col].dropna()
        out[col] = float(values.mean()) if len(values) else float("nan")
    return out


def _train(name: str, cfg: DictConfig, data: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    if name == "tabular_q":
        return train_tabular_q(data, out_dir)
    if name == "ppo" and cfg.train.source == "regime":
        return train_ppo_regime(data, out_dir)
    if name == "ppo":
        return train_ppo(data, out_dir)
    raise NotImplementedError(f"agent '{name}' is not trainable (tabular_q, ppo)")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    seed_everything(int(cfg.seed))
    chip = ChipLimits.from_dict(data["chip"])
    print(f"seed={cfg.seed} agent={cfg.agent.name} chip={chip.name} -> {out_dir}")

    name = str(cfg.agent.name)
    try:
        results = _train(name, cfg, data, out_dir)  # type: ignore[arg-type]
    except KeyboardInterrupt:
        print(f"\ninterrupted; the last checkpoint (if any) is in {out_dir / 'checkpoints'}")
        raise SystemExit(130) from None

    results.to_parquet(out_dir / "results.parquet", index=False)
    benchmarks = results["benchmark"].unique()
    for bench_name in benchmarks:
        md = "summary.md" if len(benchmarks) == 1 else f"summary_{bench_name}.md"
        write_summary_md(out_dir / md, results[results["benchmark"] == bench_name])
    reached = int((results["gap_to_optimal"] == 0).sum())
    print(
        f"\n{reached}/{len(results)} instances reached the stored optimum; "
        f"results in {out_dir / 'results.parquet'} and summary.md"
    )


if __name__ == "__main__":
    main()
