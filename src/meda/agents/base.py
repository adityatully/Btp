"""`Agent` protocol and `run_episode`: agents are evaluated exactly like solvers (SPEC 11.1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from meda.core.problem import Instance
from meda.envs.encoding import Obs
from meda.envs.sample_prep_env import SamplePrepEnv
from meda.solvers.base import SolveResult


class Agent(Protocol):
    name: str

    def act(self, obs: Obs, mask: npt.NDArray[np.bool_], deterministic: bool) -> int: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> Agent: ...


def run_episode(
    agent: Agent, env: SamplePrepEnv, instance: Instance, deterministic: bool = True
) -> SolveResult:
    """Run one episode on `instance`. Status is `feasible` on success, else `failed`.

    The result always carries the episode's DAG (complete or not) and its exact plan cost."""
    start = time.perf_counter()
    obs, _ = env.reset(options={"instance": instance})
    while True:
        action = agent.act(obs, env.action_masks(), deterministic)
        obs, _, terminated, truncated, info = env.step(np.int64(action))
        if terminated or truncated:
            break
    return SolveResult(
        instance_id=instance.instance_id,
        producer=agent.name,
        status="feasible" if info["success"] else "failed",
        dag=info["dag"].with_producer(agent.name),
        totals=info["episode_totals"],
        cost=info["plan_cost"],
        wall_time=time.perf_counter() - start,
    )
