"""PPO trainer, CleanRL style (SPEC 11.4).

Each iteration: every env takes `rollout_steps` steps with the current policy (sampled), GAE
turns the rollout into advantages (`rollout.compute_gae`), then `update_epochs` passes of
`num_minibatches` minibatches optimise the clipped surrogate, the value loss (optionally
clipped) and an entropy bonus, with gradient-norm clipping and linear learning-rate annealing.
Advantages are normalised per minibatch.

The trainer does not know where problems come from: it takes a vector env built by the caller
(fixed instances, a generator, a curriculum). The vector env must autoreset in the same step
(SPEC 9.1), so a `done` marks exactly where a new episode begins.

**Collectors.** With a vector env (lockstep), every step runs the policy once on all envs'
observations. With an `ActorPool` (`train.collector=workers`, `collector.py`), each worker plays
its env's whole rollout with a frozen copy of the weights; the learner gathers the rollouts and
the update is the same. Either way, before the first optimiser step of an update the learner's
log-probabilities and values on the first minibatch must match the collected ones (the
weights have not changed yet): a mismatch beyond rounding stops training.

**Determinism.** All randomness is the torch RNG (action sampling; with workers, one generator
per env) and the trainer's own numpy generator (minibatch order). They are saved by
`utils.checkpoint`, together with every env's snapshot, so a resumed run repeats the
uninterrupted one exactly (on CPU; GPU kernels are not bit-deterministic).
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

from meda.agents.ppo.collector import EPISODE_KEYS, ActorPool, episode_record
from meda.agents.ppo.networks import NetConfig, SetEncoderNet
from meda.agents.ppo.policy import PPOAgent, batch_obs, masked_distribution
from meda.agents.ppo.rollout import RolloutBuffer, compute_gae, split_obs
from meda.envs.encoding import EnvSizes

# The two vector envs that can read and write per-worker env snapshots (SPEC 9.1).
VecEnv = SyncVectorEnv | AsyncVectorEnv

# How far the learner's first-minibatch log-probabilities and values may be from the collected
# ones (float rounding between a CPU batch-1 pass and a padded GPU batch is ~1e-6).
CONSISTENCY_TOL = 1e-3


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
    reward_scale: float  # rewards are multiplied by this before GAE (SPEC 11.4)
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
    # Candidate rows (observations x the widest of them) per forward pass in an update. A
    # minibatch is split into chunks under this budget, widths sorted so chunks pad little;
    # gradients are accumulated, so the update is the same, only peak memory is bounded.
    max_chunk_rows: int

    def __post_init__(self) -> None:
        if self.reward_scale <= 0:
            raise ValueError(f"ppo: reward_scale must be > 0, got {self.reward_scale}")
        if self.batch_size % self.num_minibatches:
            raise ValueError(
                f"ppo: num_envs * rollout_steps = {self.batch_size} is not divisible by "
                f"num_minibatches {self.num_minibatches}"
            )
        if self.max_chunk_rows < 1:
            raise ValueError(f"ppo: max_chunk_rows must be >= 1, got {self.max_chunk_rows}")
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
        envs: VecEnv | ActorPool,
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
        self.net = SetEncoderNet(cfg.net).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.update = 0
        self.global_step = 0
        self.next_obs: dict[str, npt.NDArray[Any]] | None = None  # lockstep only
        self.started = False

    @property
    def agent(self) -> PPOAgent:
        return PPOAgent(self.net, self.device)

    @property
    def done(self) -> bool:
        return self.update >= self.cfg.num_updates

    @property
    def uses_workers(self) -> bool:
        return isinstance(self.envs, ActorPool)

    def start(self) -> None:
        """Reset every env (env i seeded `seed + i`). Not needed after a checkpoint load."""
        if isinstance(self.envs, ActorPool):
            self.envs.reset(self.seed)
        else:
            self.next_obs, _ = self.envs.reset(seed=self.seed)
        self.started = True

    # ------------------------------------------------------------------ one iteration

    def iteration(self) -> dict[str, float]:
        """Collect one rollout, update the network; return this update's metrics."""
        if not self.started:
            raise RuntimeError("call start() or load a checkpoint first")
        cfg = self.cfg
        if cfg.anneal_lr:
            frac = 1.0 - self.update / cfg.num_updates
            for group in self.optimizer.param_groups:
                group["lr"] = frac * cfg.lr
        started = time.perf_counter()
        self.net.eval()
        if isinstance(self.envs, ActorPool):
            buf, last_values, episodes, legal_counts, env_seconds = self._collect_workers()
        else:
            buf, last_values, episodes, legal_counts, env_seconds = self._collect_lockstep()
        adv, ret = compute_gae(
            buf.rewards, buf.values, buf.dones, last_values, cfg.gamma, cfg.gae_lambda
        )
        collect_seconds = time.perf_counter() - started
        stats = self._optimise(buf, adv, ret)
        update_seconds = time.perf_counter() - started - collect_seconds
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
            # Where an iteration's time goes: waiting for the env workers (with the workers
            # collector, their whole rollouts including their policy passes), everything else
            # while collecting (policy forward, copies), and the PPO update.
            "seconds_env": env_seconds,
            "seconds_policy": collect_seconds - env_seconds,
            "seconds_update": update_seconds,
        }
        return metrics

    def _collect_workers(
        self,
    ) -> tuple[RolloutBuffer, npt.NDArray[np.float64], list[dict[str, Any]], list[float], float]:
        """Every worker plays its env's rollout with this update's weights (`collector.py`)."""
        cfg = self.cfg
        assert isinstance(self.envs, ActorPool)
        waited = time.perf_counter()
        rollouts = self.envs.rollout(self.net.state_dict(), self.update, cfg.rollout_steps)
        env_seconds = time.perf_counter() - waited
        buf = RolloutBuffer(cfg.rollout_steps, cfg.num_envs)
        for t in range(cfg.rollout_steps):
            buf.add(
                [r.obs[t] for r in rollouts],
                np.array([r.actions[t] for r in rollouts]),
                np.array([r.logprobs[t] for r in rollouts]),
                np.array([r.values[t] for r in rollouts]),
                np.array([r.rewards[t] for r in rollouts]) * cfg.reward_scale,
                np.array([r.dones[t] for r in rollouts]),
            )
        self.global_step += cfg.batch_size
        legal_counts = [
            float(np.mean([r.obs[t]["action_mask"].sum() for r in rollouts]))
            for t in range(cfg.rollout_steps)
        ]
        episodes = [e for r in rollouts for e in r.episodes]
        last_values = np.array([r.last_value for r in rollouts], dtype=np.float64)
        return buf, last_values, episodes, legal_counts, env_seconds

    def _collect_lockstep(
        self,
    ) -> tuple[RolloutBuffer, npt.NDArray[np.float64], list[dict[str, Any]], list[float], float]:
        """All envs step together; one policy pass per step on their batched observations."""
        cfg = self.cfg
        assert not isinstance(self.envs, ActorPool) and self.next_obs is not None
        buf = RolloutBuffer(cfg.rollout_steps, cfg.num_envs)
        episodes: list[dict[str, Any]] = []
        legal_counts: list[float] = []
        env_seconds = 0.0
        for _ in range(cfg.rollout_steps):
            obs = self.next_obs
            with torch.no_grad():
                t = batch_obs(obs, self.device)
                logits, values = self.net(t)
                dist = masked_distribution(logits, t["action_mask"])
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
            act = actions.cpu().numpy()
            # Keep this step's observations before stepping: a vector env built with
            # `copy=False` (the training default) reuses its observation buffer.
            kept = split_obs(obs)
            legal_counts.append(float(obs["action_mask"].sum(axis=1).mean()))
            stepped = time.perf_counter()
            self.next_obs, rewards, terminated, truncated, info = self.envs.step(act)
            env_seconds += time.perf_counter() - stepped
            dones = np.logical_or(terminated, truncated)
            buf.add(
                kept,
                act,
                logprobs.cpu().numpy(),
                values.cpu().numpy(),
                rewards * cfg.reward_scale,
                dones,
            )
            episodes += _finished_episodes(info, dones)
            self.global_step += cfg.num_envs

        with torch.no_grad():
            _, last = self.net(batch_obs(self.next_obs, self.device))
        return buf, last.cpu().numpy().astype(np.float64), episodes, legal_counts, env_seconds

    def _optimise(
        self, buf: RolloutBuffer, adv: npt.NDArray[np.float64], ret: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        cfg = self.cfg
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
        widths = buf.candidate_widths()
        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        clipped = 0.0
        n = 0
        worst = {"logprob": 0.0, "value": 0.0}  # first minibatch vs collected, before any step
        for _ in range(cfg.update_epochs):
            self.rng.shuffle(index)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = index[start : start + cfg.minibatch_size]
                # Advantages are normalised over the whole minibatch, before chunking.
                adv_mean, adv_std = b_adv[mb].mean(), b_adv[mb].std()
                self.optimizer.zero_grad()
                totals = dict.fromkeys(sums, 0.0)
                clip_count = 0.0
                for chunk in chunk_by_width(mb, widths, cfg.max_chunk_rows):
                    share = len(chunk) / len(mb)
                    t = batch_obs(buf.minibatch_obs(chunk), self.device)
                    logits, new_values = self.net(t)
                    dist = masked_distribution(logits, t["action_mask"])
                    new_logprobs = dist.log_prob(b_actions[chunk])
                    entropy = dist.entropy().mean()
                    log_ratio = new_logprobs - b_logprobs[chunk]
                    ratio = log_ratio.exp()
                    if n == 0:
                        with torch.no_grad():
                            v_diff = (new_values - b_values[chunk]).abs() / (
                                1 + b_values[chunk].abs()
                            )
                            worst["logprob"] = max(worst["logprob"], log_ratio.abs().max().item())
                            worst["value"] = max(worst["value"], v_diff.max().item())
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_count += float(((ratio - 1).abs() > cfg.clip_coef).float().sum())

                    c_adv = (b_adv[chunk] - adv_mean) / (adv_std + 1e-8)
                    pg_loss = torch.max(
                        -c_adv * ratio,
                        -c_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),
                    ).mean()
                    if cfg.clip_vloss:
                        v_clip = b_values[chunk] + (new_values - b_values[chunk]).clamp(
                            -cfg.clip_coef, cfg.clip_coef
                        )
                        v_loss = (
                            0.5
                            * torch.max(
                                (new_values - b_ret[chunk]) ** 2, (v_clip - b_ret[chunk]) ** 2
                            ).mean()
                        )
                    else:
                        v_loss = 0.5 * ((new_values - b_ret[chunk]) ** 2).mean()
                    loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss
                    # Each chunk's mean weighted by its share: the minibatch mean, exactly.
                    (share * loss).backward()

                    totals["policy_loss"] += share * pg_loss.item()
                    totals["value_loss"] += share * v_loss.item()
                    totals["entropy"] += share * entropy.item()
                    totals["approx_kl"] += share * approx_kl.item()

                if n == 0 and max(worst.values()) > CONSISTENCY_TOL:
                    raise RuntimeError(
                        f"update {self.update}: the learner's log-probabilities or values on the "
                        f"first minibatch differ from the collected ones by up to "
                        f"{worst['logprob']:.3g} (log-prob) and {worst['value']:.3g} (relative "
                        f"value), above {CONSISTENCY_TOL}: the policy that collected the rollout "
                        f"is not the one being updated"
                    )
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                for k in sums:
                    sums[k] += totals[k]
                clipped += clip_count / len(mb)
                n += 1

        y_true, y_pred = b_ret.cpu().numpy(), b_values.cpu().numpy()
        var = float(np.var(y_true))
        explained = float("nan") if var == 0 else 1.0 - float(np.var(y_true - y_pred)) / var
        return {
            **{k: v / n for k, v in sums.items()},
            "collected_logprob_max_diff": worst["logprob"],
            "collected_value_max_diff": worst["value"],
            "clip_fraction": clipped / n,
            "explained_variance": explained,
        }


def chunk_by_width(
    mb: npt.NDArray[np.int64], widths: npt.NDArray[np.int64], max_rows: int
) -> list[npt.NDArray[np.int64]]:
    """`mb` split into chunks, sorted by candidate width, each holding at most `max_rows`
    padded candidate rows (len x widest); a single observation wider than the budget is a
    chunk of its own."""
    order = mb[np.argsort(widths[mb], kind="stable")]
    chunks: list[npt.NDArray[np.int64]] = []
    begin = 0
    for end in range(1, len(order) + 1):
        if end < len(order) and (end + 1 - begin) * int(widths[order[end]]) > max_rows:
            chunks.append(order[begin:end])
            begin = end
    chunks.append(order[begin:])
    return [c for c in chunks if len(c)]


def _finished_episodes(
    info: Mapping[str, Any], dones: npt.NDArray[np.bool_]
) -> list[dict[str, Any]]:
    """Per finished episode: the env's end-of-episode info (SAME_STEP puts it in final_info)."""
    if not dones.any():
        return []
    final = info["final_info"]
    return [episode_record({k: final[k][i] for k in EPISODE_KEYS}) for i in np.flatnonzero(dones)]


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
