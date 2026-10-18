"""Rollout buffer and GAE (SPEC 11.4).

**When to bootstrap.** An episode that ended at step t (success, truncation or dead end) is
never bootstrapped: the env already charged the rest of the plan as the `cost_to_finish`
penalty (SPEC 9.1), so adding V(s') would count it twice. The one bootstrap is at the end of a
rollout, for episodes the rollout cut off mid-way: their remaining cost is estimated by the
value of the next observation. The vector env autoresets in the same step (SPEC 9.1), so
`dones[t]` marks exactly the steps after which the next observation belongs to a new episode.

**Memory.** Observations are stored per step, trimmed to that step's longest real candidate
list, and padded only to the widest step of the rollout when the batch is built. A rollout at
`L_max = 2048` would otherwise hold gigabytes of padding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from meda.agents.ppo.policy import CANDIDATE_KEYS, trim_candidates

FloatArray = npt.NDArray[np.float64]


def compute_gae(
    rewards: FloatArray,
    values: FloatArray,
    dones: npt.NDArray[np.bool_],
    last_values: FloatArray,
    gamma: float,
    lam: float,
) -> tuple[FloatArray, FloatArray]:
    """Advantages and returns, both [T, N].

    `dones[t, i]`: env i's episode ended with the action at step t (no bootstrap through it).
    `last_values[i]`: V of env i's observation after the last step (the cut-off bootstrap).
    """
    steps = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    running = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in reversed(range(steps)):
        nxt = last_values if t == steps - 1 else values[t + 1]
        live = 1.0 - dones[t].astype(np.float64)
        delta = rewards[t] + gamma * nxt * live - values[t]
        running = delta + gamma * lam * live * running
        adv[t] = running
    return adv, adv + values


class RolloutBuffer:
    """One rollout of `num_envs` envs: per-step observations and the PPO quantities."""

    def __init__(self, num_steps: int, num_envs: int) -> None:
        self.num_steps, self.num_envs = num_steps, num_envs
        self.obs: list[dict[str, npt.NDArray[Any]]] = []
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float64)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float64)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float64)
        self.dones = np.zeros((num_steps, num_envs), dtype=bool)

    def add(
        self,
        obs: Mapping[str, npt.NDArray[Any]],
        actions: npt.NDArray[np.int64],
        logprobs: FloatArray,
        values: FloatArray,
        rewards: FloatArray,
        dones: npt.NDArray[np.bool_],
    ) -> None:
        t = len(self.obs)
        if t >= self.num_steps:
            raise IndexError(f"rollout buffer is full ({self.num_steps} steps)")
        self.obs.append(trim_candidates(obs))
        self.actions[t], self.logprobs[t], self.values[t] = actions, logprobs, values
        self.rewards[t], self.dones[t] = rewards, dones

    def flat_obs(self) -> dict[str, npt.NDArray[Any]]:
        """All observations as [T * N, ...], candidates padded to the widest step."""
        if len(self.obs) != self.num_steps:
            raise ValueError(f"rollout has {len(self.obs)} of {self.num_steps} steps")
        width = max(o["action_mask"].shape[1] for o in self.obs)
        out: dict[str, npt.NDArray[Any]] = {}
        for key in self.obs[0]:
            parts = [o[key] for o in self.obs]
            if key in CANDIDATE_KEYS:
                fill = -1 if key == "cand_index" else 0
                parts = [_pad_width(p, width, fill) for p in parts]
            out[key] = np.concatenate(parts, axis=0)
        return out


def _pad_width(a: npt.NDArray[Any], width: int, fill: int) -> npt.NDArray[Any]:
    if a.shape[1] == width:
        return a
    pad = [(0, 0), (0, width - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pad, constant_values=fill)
