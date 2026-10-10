"""Report env steps/second (SPEC 9.4).

Usage:
    uv run python scripts/profile_env.py generator=tiny_dilution
    uv run python scripts/profile_env.py generator=dilution_multi +steps=50000 +workers=16

Instances come from the generator stage itself (the real p(s0), SPEC 8.1), drawn once up front
with the run's seed. Actions are uniformly random legal candidates. With `+workers=N` (N > 0)
the run uses `gymnasium.vector.AsyncVectorEnv` with N processes and reports the total rate.
The truncation penalty is 0 here: the real estimate is `solvers.fallback` (M5), and it would
only change rewards, not the step rate.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from meda.core.chip import ChipLimits
from meda.core.composition import ZERO, Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.core.state import PrepState
from meda.envs.sample_prep_env import EnvConfig, FixedInstances, SamplePrepEnv
from meda.generators.sampler import GeneratorConfig, sample_instances


def zero_penalty(state: PrepState, instance: Instance) -> Rational:
    """Stand-in until `solvers.fallback` (M5)."""
    return ZERO


def make_instances(cfg: dict[str, Any], count: int, seed: int) -> list[Instance]:
    return sample_instances(
        GeneratorConfig.from_dict(cfg["generator"]),
        ChipLimits.from_dict(cfg["chip"]),
        np.random.default_rng(seed),
        count=count,
        weights=CostWeights.from_dict(cfg["costs"]),
        unique=False,  # a speed measurement does not need distinct instances
    )


def make_env(cfg: dict[str, Any], instances: list[Instance]) -> SamplePrepEnv:
    return SamplePrepEnv(
        EnvConfig.from_dict(cfg["env"]),
        FixedInstances(instances),
        CostWeights.from_dict(cfg["costs"]),
        zero_penalty,
    )


def profile_single(cfg: dict[str, Any], instances: list[Instance], steps: int, seed: int) -> None:
    env = make_env(cfg, instances)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    episodes, max_legal = 0, 0
    start = time.perf_counter()
    for _ in range(steps):
        legal = len(env.legal)
        max_legal = max(max_legal, legal)
        _, _, terminated, truncated, _ = env.step(np.int64(rng.integers(legal)))
        if terminated or truncated:
            episodes += 1
            env.reset()
    dt = time.perf_counter() - start
    print(
        f"single process: {steps / dt:,.0f} steps/s ({steps} steps, {episodes} episodes, "
        f"max legal candidates {max_legal}, L_max {env.config.sizes.l_max})"
    )


def profile_vector(
    cfg: dict[str, Any], instances: list[Instance], steps: int, workers: int, seed: int
) -> None:
    from gymnasium.vector import AsyncVectorEnv

    venv = AsyncVectorEnv([partial(make_env, cfg, instances) for _ in range(workers)])
    rng = np.random.default_rng(seed)
    obs, _ = venv.reset(seed=seed)
    start = time.perf_counter()
    rounds = max(1, steps // workers)
    for _ in range(rounds):
        actions = []
        for mask in obs["action_mask"]:
            legal = np.flatnonzero(mask)
            actions.append(int(rng.choice(legal)) if legal.size else 0)  # 0: autoreset step
        obs, *_ = venv.step(np.array(actions))
    dt = time.perf_counter() - start
    venv.close()
    print(f"{workers} workers: {rounds * workers / dt:,.0f} steps/s total")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    steps = int(cfg.get("steps", 20_000))
    workers = int(cfg.get("workers", 0))
    seed = int(cfg.seed)
    instances = make_instances(data, int(cfg.get("instances", 32)), seed)  # type: ignore[arg-type]
    limits = [i.step_limit for i in instances]
    print(
        f"generator={cfg.generator.name} chip={cfg.chip.name}: {len(instances)} sampled "
        f"instances, step limits {min(limits)}-{max(limits)}; random legal actions"
    )
    profile_single(data, instances, steps, seed)  # type: ignore[arg-type]
    if workers > 0:
        profile_vector(data, instances, steps, workers, seed)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
