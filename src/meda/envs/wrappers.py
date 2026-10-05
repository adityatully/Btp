"""`EpisodeStats`, `CurriculumWrapper`, `PreferenceWrapper` (SPEC 9.5)."""

from __future__ import annotations

from typing import Any, Protocol, SupportsFloat

import gymnasium as gym
import numpy as np

from meda.envs.encoding import Obs
from meda.envs.sample_prep_env import InstanceSource, SamplePrepEnv


class EpisodeStats(gym.Wrapper[Obs, np.int64, Obs, np.int64]):
    """Adds `info["episode_stats"]` at episode end: return, length, status and cost components."""

    def __init__(self, env: gym.Env[Obs, np.int64]) -> None:
        super().__init__(env)
        self._return = 0.0
        self._length = 0

    def reset(self, **kwargs: Any) -> tuple[Obs, dict[str, Any]]:
        self._return, self._length = 0.0, 0
        return self.env.reset(**kwargs)

    def step(self, action: np.int64) -> tuple[Obs, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._return += float(reward)
        self._length += 1
        if terminated or truncated:
            totals = info["episode_totals"]
            info["episode_stats"] = {
                "return": self._return,
                "length": self._length,
                "success": info["success"],
                "dead_end": info["dead_end"],
                "truncated": truncated,
                "plan_cost": float(info["plan_cost"]),
                "penalty": float(info["penalty"]),
                "mix_ops": totals.mix_ops,
                "waste_volume": totals.waste_volume,
                "total_reagent": totals.total_reagent,
                "reagent_by_fluid": totals.reagent_by_fluid,
            }
        return obs, reward, terminated, truncated, info


class Curriculum(Protocol):
    """Implemented in `meda.generators.curriculum` (M4)."""

    def source(self) -> InstanceSource: ...

    def record(self, episode_info: dict[str, Any]) -> None: ...


class CurriculumWrapper(gym.Wrapper[Obs, np.int64, Obs, np.int64]):
    """Reports finished episodes to the curriculum and draws each new episode from its current
    stage."""

    def __init__(self, env: gym.Env[Obs, np.int64], curriculum: Curriculum) -> None:
        super().__init__(env)
        self.curriculum = curriculum

    def reset(self, **kwargs: Any) -> tuple[Obs, dict[str, Any]]:
        base = self.env.unwrapped
        assert isinstance(base, SamplePrepEnv)
        base.set_instance_source(self.curriculum.source())
        return self.env.reset(**kwargs)

    def step(self, action: np.int64) -> tuple[Obs, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.curriculum.record(info)
        return obs, reward, terminated, truncated, info


class PreferenceWrapper(gym.Wrapper[Obs, np.int64, Obs, np.int64]):
    """Reserved (SPEC 6.5, milestone 9): sample cost weights per episode."""

    def __init__(self, env: gym.Env[Obs, np.int64]) -> None:
        raise NotImplementedError("preference conditioning is reserved for milestone 9")
