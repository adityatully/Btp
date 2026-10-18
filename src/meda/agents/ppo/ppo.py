"""PPO trainer, CleanRL style (SPEC 11.4).

Each iteration: every env takes `rollout_steps` steps with the current policy (sampled), GAE
turns the rollout into advantages (`rollout.compute_gae`), then `update_epochs` passes of
`num_minibatches` minibatches optimise the clipped surrogate, the value loss (optionally
clipped) and an entropy bonus, with gradient-norm clipping and linear learning-rate annealing.
Advantages are normalised per minibatch.

The trainer does not know where problems come from: it takes a vector env built by the caller
(fixed instances, a generator, a curriculum). The vector env must autoreset in the same step
(SPEC 9.1), so a `done` marks exactly where a new episode begins.

**Determinism.** All randomness is the torch RNG (action sampling) and the trainer's own numpy
generator (minibatch order). Both are saved by `utils.checkpoint`, together with every env's
snapshot, so a resumed run repeats the uninterrupted one exactly (on CPU; GPU kernels are not
bit-deterministic).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch import nn

from meda.agents.ppo.networks import NetConfig, SetEncoderNet
from meda.agents.ppo.policy import PPOAgent, batch_obs, masked_distribution
from meda.agents.ppo.rollout import RolloutBuffer, compute_gae
from meda.envs.encoding import EnvSizes

# The two vector envs that can read and write per-worker env snapshots (SPEC 9.1).
VecEnv = SyncVectorEnv | AsyncVectorEnv


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """`configs/agent/ppo.yaml`. Unknown or missing keys raise."""

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    num_envs: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    clip_vloss: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    lr: float
    anneal_lr: bool
    update_epochs: int
    num_minibatches: int
    total_env_steps: int

    def __post_init__(self) -> None:
        if self.batch_size % self.num_minibatches:
            raise ValueError(
                f"ppo: num_envs * rollout_steps = {self.batch_size} is not divisible by "
                f"num_minibatches {self.num_minibatches}"
            )
        if self.num_updates < 1:
            raise ValueError(
                f"ppo: total_env_steps {self.total_env_steps} < one rollout ({self.batch_size})"
            )

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_env_steps // self.batch_size

    @property
    def net(self) -> NetConfig:
        return NetConfig(self.d_model, self.n_layers, self.n_heads)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> PPOConfig:
        names = {f.name for f in fields(PPOConfig)}
        if set(data) != names:
            raise ValueError(
                f"ppo config keys mismatch: missing {sorted(names - set(data))}, "
                f"unknown {sorted(set(data) - names)}"
            )
        types = {f.name: f.type for f in fields(PPOConfig)}
        cast = {"int": int, "float": float, "bool": bool, "str": str}
        return PPOConfig(**{k: cast[str(types[k])](data[k]) for k in names})


class PPOTrainer:
    def __init__(
        self,
        cfg: PPOConfig,
        sizes: EnvSizes,
        envs: VecEnv,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if envs.num_envs != cfg.num_envs:
            raise ValueError(f"vector env has {envs.num_envs} envs, config says {cfg.num_envs}")
        mode = envs.metadata.get("autoreset_mode")
        if mode is None or getattr(mode, "value", mode) != "SameStep":
            raise ValueError(f"PPO needs AutoresetMode.SAME_STEP (SPEC 9.1), got {mode}")
        self.cfg, self.sizes, self.envs = cfg, sizes, envs
        self.device = torch.device(device)
        self.net = SetEncoderNet(sizes, cfg.net).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.update = 0
        self.global_step = 0
        self.next_obs: dict[str, npt.NDArray[Any]] | None = None

    @property
    def agent(self) -> PPOAgent:
        return PPOAgent(self.net, self.device)

    @property
    def done(self) -> bool:
        return self.update >= self.cfg.num_updates

    def start(self) -> None:
        """Reset every env (env i seeded `seed + i`). Not needed after a checkpoint load."""
        self.next_obs, _ = self.envs.reset(seed=self.seed)

    # ------------------------------------------------------------------ one iteration

    def iteration(self) -> dict[str, float]:
        """Collect one rollout, update the network; return this update's metrics."""
        if self.next_obs is None:
            raise RuntimeError("call start() or load a checkpoint first")
        cfg = self.cfg
        if cfg.anneal_lr:
            frac = 1.0 - self.update / cfg.num_updates
            for group in self.optimizer.param_groups:
                group["lr"] = frac * cfg.lr
        started = time.perf_counter()
        buf = RolloutBuffer(cfg.rollout_steps, cfg.num_envs)
        episodes: list[dict[str, Any]] = []
        legal_counts: list[float] = []

        self.net.eval()
        for _ in range(cfg.rollout_steps):
            obs = self.next_obs
            with torch.no_grad():
                t = batch_obs(obs, self.device)
                logits, values = self.net(t)
                dist = masked_distribution(logits, t["action_mask"])
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
            act = actions.cpu().numpy()
            self.next_obs, rewards, terminated, truncated, info = self.envs.step(act)
            dones = np.logical_or(terminated, truncated)
            buf.add(obs, act, logprobs.cpu().numpy(), values.cpu().numpy(), rewards, dones)
            legal_counts.append(float(obs["action_mask"].sum(axis=1).mean()))
            episodes += _finished_episodes(info, dones)
            self.global_step += cfg.num_envs

        with torch.no_grad():
            _, last = self.net(batch_obs(self.next_obs, self.device))
        adv, ret = compute_gae(
            buf.rewards, buf.values, buf.dones, last.cpu().numpy(), cfg.gamma, cfg.gae_lambda
        )
        collect_seconds = time.perf_counter() - started
        stats = self._optimise(buf, adv, ret)
        self.update += 1

        metrics = {
            "update": float(self.update),
            "global_step": float(self.global_step),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "legal_count": float(np.mean(legal_counts)),
            "episodes": float(len(episodes)),
            **_episode_metrics(episodes),
            **stats,
            "env_steps_per_second": cfg.batch_size / collect_seconds,
        }
        return metrics

    def _optimise(
        self, buf: RolloutBuffer, adv: npt.NDArray[np.float64], ret: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        cfg = self.cfg
        obs = buf.flat_obs()
        b_actions = torch.as_tensor(buf.actions.reshape(-1), device=self.device)
        b_logprobs = torch.as_tensor(buf.logprobs.reshape(-1), dtype=torch.float32)
        b_values = torch.as_tensor(buf.values.reshape(-1), dtype=torch.float32)
        b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
        b_ret = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
        b_logprobs, b_values, b_adv, b_ret = (
            x.to(self.device) for x in (b_logprobs, b_values, b_adv, b_ret)
        )

        self.net.train()
        index = np.arange(cfg.batch_size)
        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        clipped = 0.0
        n = 0
        for _ in range(cfg.update_epochs):
            self.rng.shuffle(index)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = index[start : start + cfg.minibatch_size]
                t = batch_obs({k: v[mb] for k, v in obs.items()}, self.device)
                logits, new_values = self.net(t)
                dist = masked_distribution(logits, t["action_mask"])
                new_logprobs = dist.log_prob(b_actions[mb])
                entropy = dist.entropy().mean()
                log_ratio = new_logprobs - b_logprobs[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipped += float(((ratio - 1).abs() > cfg.clip_coef).float().mean())

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()
                if cfg.clip_vloss:
                    v_clip = b_values[mb] + (new_values - b_values[mb]).clamp(
                        -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max((new_values - b_ret[mb]) ** 2, (v_clip - b_ret[mb]) ** 2).mean()
                    )
                else:
                    v_loss = 0.5 * ((new_values - b_ret[mb]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                sums["policy_loss"] += pg_loss.item()
                sums["value_loss"] += v_loss.item()
                sums["entropy"] += entropy.item()
                sums["approx_kl"] += approx_kl.item()
                n += 1

        y_true, y_pred = b_ret.cpu().numpy(), b_values.cpu().numpy()
        var = float(np.var(y_true))
        explained = float("nan") if var == 0 else 1.0 - float(np.var(y_true - y_pred)) / var
        return {
            **{k: v / n for k, v in sums.items()},
            "clip_fraction": clipped / n,
            "explained_variance": explained,
        }


def _finished_episodes(
    info: Mapping[str, Any], dones: npt.NDArray[np.bool_]
) -> list[dict[str, Any]]:
    """Per finished episode: the env's end-of-episode info (SAME_STEP puts it in final_info)."""
    if not dones.any():
        return []
    final = info["final_info"]
    out = []
    for i in np.flatnonzero(dones):
        totals = final["episode_totals"][i]
        out.append(
            {
                "success": bool(final["success"][i]),
                "dead_end": bool(final["dead_end"][i]),
                "plan_cost": float(final["plan_cost"][i]),
                "penalty": float(final["penalty"][i]),
                "mix_ops": totals.mix_ops,
                "waste_volume": totals.waste_volume,
                "total_reagent": totals.total_reagent,
            }
        )
    return out


def _episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    """SPEC 11.4: return, success and dead-end rates, cost and its components (successes)."""
    nan = float("nan")
    if not episodes:
        return dict.fromkeys(
            (
                "episode_return",
                "success_rate",
                "dead_end_rate",
                "plan_cost",
                "mix_ops",
                "waste_volume",
                "total_reagent",
            ),
            nan,
        )
    wins = [e for e in episodes if e["success"]]

    def mean_over_wins(key: str) -> float:
        return float(np.mean([e[key] for e in wins])) if wins else nan

    return {
        "episode_return": float(np.mean([-(e["plan_cost"] + e["penalty"]) for e in episodes])),
        "success_rate": len(wins) / len(episodes),
        "dead_end_rate": float(np.mean([e["dead_end"] for e in episodes])),
        "plan_cost": mean_over_wins("plan_cost"),
        "mix_ops": mean_over_wins("mix_ops"),
        "waste_volume": mean_over_wins("waste_volume"),
        "total_reagent": mean_over_wins("total_reagent"),
    }
