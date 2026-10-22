"""Rollout buffer and GAE (SPEC 11.4).

**When to bootstrap.** An episode that ended at step t (success, truncation or dead end) is
never bootstrapped: the env already charged the rest of the plan as the `cost_to_finish`
penalty (SPEC 9.1), so adding V(s') would count it twice. The one bootstrap is at the end of a
rollout, for episodes the rollout cut off mid-way: their remaining cost is estimated by the
value of the next observation. The vector env autoresets in the same step (SPEC 9.1), so
`dones[t]` marks exactly the steps after which the next observation belongs to a new episode.

**Memory.** Observations are stored per step and per env, every token axis trimmed to that
observation's own real widths, and padded only to the widest observation of a minibatch when
the minibatch is built. Candidate counts vary widely (tens on an empty chip, thousands at
M = 10 with many droplets), so padding a whole rollout to its widest step would hold
gigabytes; the stored rollout grows with the mean width and a minibatch with its own widest.
Each stored observation is a **copy** (`split_obs`): a trimmed view would keep the whole padded
batch it came from alive (measured: 21 GB at 32 envs and L_pad 16384), and a vector env built
with `copy=False` overwrites that batch at its next step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from meda.agents.ppo.policy import KEY_AXES, PAD_FILL, real_widths, trim_obs

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


def split_obs(obs: Mapping[str, npt.NDArray[Any]]) -> list[dict[str, npt.NDArray[Any]]]:
    """A batched observation as one trimmed, owned copy per env (row i of every key)."""
    rows = len(next(iter(obs.values())))
    return [
        {k: np.array(v) for k, v in trim_obs({k: v[i : i + 1] for k, v in obs.items()}).items()}
        for i in range(rows)
    ]


class RolloutBuffer:
    """One rollout of `num_envs` envs: per-step observations and the PPO quantities."""

    def __init__(self, num_steps: int, num_envs: int) -> None:
        self.num_steps, self.num_envs = num_steps, num_envs
        self.obs: list[dict[str, npt.NDArray[Any]]] = []  # flat index t * num_envs + env
        self.steps_added = 0
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float64)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float64)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float64)
        self.dones = np.zeros((num_steps, num_envs), dtype=bool)

    def add(
        self,
        obs: Mapping[str, npt.NDArray[Any]] | list[dict[str, npt.NDArray[Any]]],
        actions: npt.NDArray[np.int64],
        logprobs: FloatArray,
        values: FloatArray,
        rewards: FloatArray,
        dones: npt.NDArray[np.bool_],
    ) -> None:
        """`obs` is the batched observation the actions were taken in, or `split_obs` of it
        (taken before the vector env stepped, when it reuses its observation buffer)."""
        t = self.steps_added
        if t >= self.num_steps:
            raise IndexError(f"rollout buffer is full ({self.num_steps} steps)")
        kept = obs if isinstance(obs, list) else split_obs(obs)
        if len(kept) != self.num_envs:
            raise ValueError(f"{len(kept)} observations for {self.num_envs} envs")
        self.obs.extend(kept)
        self.steps_added += 1
        self.actions[t], self.logprobs[t], self.values[t] = actions, logprobs, values
        self.rewards[t], self.dones[t] = rewards, dones

    def minibatch_obs(self, index: npt.NDArray[np.int64]) -> dict[str, npt.NDArray[Any]]:
        """The observations at flat indices `index` (t * num_envs + env) as [len(index), ...],
        each token axis padded to the widest of them."""
        if self.steps_added != self.num_steps:
            raise ValueError(f"rollout has {self.steps_added} of {self.num_steps} steps")
        chosen = [self.obs[int(i)] for i in index]
        each = [real_widths(o) for o in chosen]
        widths = {axis: max(w[axis] for w in each) for axis in each[0]}
        out: dict[str, npt.NDArray[Any]] = {}
        for key in chosen[0]:
            fill = PAD_FILL.get(key, 0)
            parts = [_pad_to(o[key], key, widths, fill) for o in chosen]
            out[key] = np.concatenate(parts, axis=0)
        return out

    def candidate_widths(self) -> npt.NDArray[np.int64]:
        """Each stored observation's candidate row count (its trimmed width)."""
        return np.array([o["action_mask"].shape[1] for o in self.obs], dtype=np.int64)

    def flat_obs(self) -> dict[str, npt.NDArray[Any]]:
        """All observations as [T * N, ...], each token axis padded to the widest."""
        return self.minibatch_obs(np.arange(self.num_steps * self.num_envs))


def _pad_to(a: npt.NDArray[Any], key: str, widths: dict[str, int], fill: int) -> npt.NDArray[Any]:
    pad = [(0, 0)] * a.ndim
    for dim, axis in KEY_AXES[key]:
        pad[dim] = (0, widths[axis] - a.shape[dim])
    if all(p == (0, 0) for p in pad):
        return a
    return np.pad(a, pad, constant_values=fill)
