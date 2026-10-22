"""Worker-side rollout collection (`train.collector=workers`, SPEC 11.4).

The lockstep collector steps all envs together and runs the policy once per step on the
batch, so every step waits for the slowest env (env steps vary 10x with the number of legal
actions). Here each worker process owns one env **and a copy of the policy**: it plays its
whole rollout of `rollout_steps` actions alone, then the learner gathers all of them. This is
synchronous on-policy PPO with decentralised actors (as DD-PPO): there is no policy lag.

Safety, in order of the update cycle:
- **Frozen weights per rollout.** The learner sends the same weights, stamped with the update
  number, to every worker; workers never change them, and the learner changes them only after
  every rollout of that update is back. A rollout with another stamp is refused.
- **No shared mutable state.** A worker touches only its own env, network copy and RNG.
- **Determinism (rule 6).** Each env's actions are sampled with its own `torch.Generator`
  (seeded from the run seed and the env index), from a batch-1 forward pass on one CPU thread,
  so results do not depend on timing. The generators are saved in checkpoints.
- **Failures stop the run.** A worker error is raised in the learner with the worker's
  traceback; a dead worker is an error, never a smaller batch.
- **Ctrl-C** is ignored by workers; the learner terminates them (`close(terminate=True)`).

The learner re-checks the workers' log-probabilities and values on its first minibatch
(`PPOTrainer`), so a mismatch between the two copies of the network stops training.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import torch
from gymnasium.vector import AutoresetMode

from meda.agents.ppo.networks import NetConfig, SetEncoderNet
from meda.agents.ppo.policy import batch_obs, masked_distribution
from meda.agents.ppo.rollout import split_obs
from meda.envs.sample_prep_env import SamplePrepEnv

EPISODE_KEYS = ("success", "dead_end", "plan_cost", "penalty", "episode_totals")
# Env methods `ActorPool.call` may run: none of them moves an episode on, so the actor's
# current observation stays the env's own (a raw reset or step would desynchronise them).
CALLABLE = frozenset({"set_instance_source", "observe"})


def episode_record(end: Mapping[str, Any]) -> dict[str, Any]:
    """An ended episode's metrics from its last step's info (SPEC 11.4)."""
    totals = end["episode_totals"]
    return {
        "success": bool(end["success"]),
        "dead_end": bool(end["dead_end"]),
        "plan_cost": float(end["plan_cost"]),
        "penalty": float(end["penalty"]),
        "mix_ops": totals.mix_ops,
        "waste_volume": totals.waste_volume,
        "total_reagent": totals.total_reagent,
    }


@dataclass
class EnvRollout:
    """One env's rollout: per step the observation acted in (trimmed copy), action, its
    log-probability and value, reward and done; the value after the last step; ended episodes."""

    version: int
    obs: list[dict[str, npt.NDArray[Any]]]
    actions: npt.NDArray[np.int64]
    logprobs: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    rewards: npt.NDArray[np.float64]
    dones: npt.NDArray[np.bool_]
    last_value: float
    episodes: list[dict[str, Any]]


def actor_seed(seed: int, index: int) -> int:
    """The sampling seed of env `index` in a run seeded `seed`."""
    return int(np.random.SeedSequence([seed, index]).generate_state(1, np.uint64)[0] >> 1)


class _Actor:
    """The worker side: one env, a policy copy, and the env's sampling generator."""

    def __init__(self, env: SamplePrepEnv, net_cfg: NetConfig) -> None:
        self.env = env
        self.net = SetEncoderNet(net_cfg).eval()
        self.gen = torch.Generator()
        self.obs: dict[str, npt.NDArray[Any]] | None = None

    def reset(self, seed: int, sampling_seed: int) -> None:
        self.obs, _ = self.env.reset(seed=seed)
        self.gen.manual_seed(sampling_seed)

    def act(self) -> tuple[dict[str, npt.NDArray[Any]], int, float, float]:
        """The current observation (trimmed copy), a sampled action, its log-prob and value."""
        assert self.obs is not None
        kept = split_obs({k: v[None] for k, v in self.obs.items()})[0]
        with torch.no_grad():
            t = batch_obs(kept, torch.device("cpu"))
            logits, value = self.net(t)
            dist = masked_distribution(logits, t["action_mask"])
            action = torch.multinomial(dist.probs, 1, generator=self.gen)[:, 0]
            logprob = dist.log_prob(action)
        a = int(action.item())
        if not self.obs["action_mask"][a]:
            raise RuntimeError(f"sampled masked candidate {a}")  # probability 0: cannot happen
        return kept, a, float(logprob.item()), float(value.item())

    def rollout(self, weights: Mapping[str, torch.Tensor], version: int, steps: int) -> EnvRollout:
        if self.obs is None:
            raise RuntimeError("actor not reset")
        self.net.load_state_dict(weights)
        obs: list[dict[str, npt.NDArray[Any]]] = []
        actions = np.zeros(steps, dtype=np.int64)
        logprobs = np.zeros(steps, dtype=np.float64)
        values = np.zeros(steps, dtype=np.float64)
        rewards = np.zeros(steps, dtype=np.float64)
        dones = np.zeros(steps, dtype=bool)
        episodes = []
        for t in range(steps):
            kept, actions[t], logprobs[t], values[t] = self.act()
            obs.append(kept)
            self.obs, rewards[t], terminated, truncated, info = self.env.step(actions[t])
            dones[t] = terminated or truncated
            if dones[t]:  # same-step autoreset (SPEC 9.1), as the lockstep vector env does
                episodes.append(episode_record(info))
                self.obs, _ = self.env.reset()
        with torch.no_grad():
            batched = {k: v[None] for k, v in self.obs.items()}
            last = self.net(batch_obs(batched, torch.device("cpu")))[1]
        return EnvRollout(
            version, obs, actions, logprobs, values, rewards, dones, float(last.item()), episodes
        )

    def get(self, name: str) -> Any:
        if name == "actor_rng":
            return self.gen.get_state()
        return getattr(self.env, name)

    def set(self, name: str, value: Any) -> None:
        if name == "actor_rng":
            self.gen.set_state(value)
            return
        setattr(self.env, name, value)
        if name == "env_snapshot":  # a restored env continues from its own observation
            self.obs = self.env.observe()


def _worker(
    remote: Connection,
    parent: Connection,
    env_fn: Callable[[], SamplePrepEnv],
    net_cfg: NetConfig,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the learner decides when workers stop
    parent.close()
    torch.set_num_threads(1)
    try:
        actor = _Actor(env_fn(), net_cfg)
    except Exception:
        remote.send(("error", traceback.format_exc()))
        return
    remote.send(("ok", None))
    while True:
        try:
            cmd, args = remote.recv()
        except EOFError:
            return
        if cmd == "close":
            remote.send(("ok", None))
            return
        try:
            out: Any = None
            if cmd == "reset":
                actor.reset(*args)
            elif cmd == "rollout":
                out = actor.rollout(*args)
            elif cmd == "call":
                name, call_args = args
                if name not in CALLABLE:
                    raise ValueError(f"env method {name!r} cannot be called through the pool")
                out = getattr(actor.env, name)(*call_args)
            elif cmd == "get":
                out = actor.get(args)
            elif cmd == "set":
                actor.set(*args)
            else:
                raise ValueError(f"unknown command {cmd!r}")
            remote.send(("ok", out))
        except Exception:
            remote.send(("error", traceback.format_exc()))


class ActorPool:
    """`num_envs` worker processes, each with one env and a policy copy (module docstring).

    Offers the part of the vector-env interface the trainer, checkpoints and the regime use:
    `num_envs`, `metadata`, `reset`, `call`, `get_attr`, `set_attr`, `close`, plus `rollout`.
    Workers are started with `spawn` (forking a process that has initialised CUDA is unsafe),
    so each env factory must be picklable (e.g. a `functools.partial` of a module function)."""

    metadata: ClassVar[dict[str, Any]] = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, env_fns: Sequence[Callable[[], SamplePrepEnv]], net_cfg: NetConfig) -> None:
        ctx = mp.get_context("spawn")
        self.num_envs = len(env_fns)
        self.closed = False
        self._remotes: list[Connection] = []
        self._procs: list[Any] = []
        for i, fn in enumerate(env_fns):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child, parent, fn, net_cfg),
                name=f"meda-actor-{i}",
                daemon=True,
            )
            proc.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(proc)
        self._gather()

    def _gather(self) -> list[Any]:
        """Every worker's reply, in env order; any error stops all workers and is raised."""
        out, errors = [], []
        for i, remote in enumerate(self._remotes):
            try:
                status, value = remote.recv()
            except (EOFError, ConnectionResetError):
                status, value = "error", "worker process died"
            if status == "error":
                errors.append(f"env worker {i}:\n{value}")
            out.append(value)
        if errors:
            self.close(terminate=True)
            raise RuntimeError("\n".join(errors))
        return out

    def _all(self, cmd: str, args: Sequence[Any]) -> list[Any]:
        if self.closed:
            raise RuntimeError("actor pool is closed")
        for remote, a in zip(self._remotes, args, strict=True):
            remote.send((cmd, a))
        return self._gather()

    def reset(self, seed: int) -> None:
        """Env i is reset with seed `seed + i` (as a vector env) and samples with its own
        generator seeded `actor_seed(seed, i)`."""
        self._all("reset", [(seed + i, actor_seed(seed, i)) for i in range(self.num_envs)])

    def rollout(
        self, weights: Mapping[str, torch.Tensor], version: int, steps: int
    ) -> list[EnvRollout]:
        cpu = {k: v.detach().to("cpu") for k, v in weights.items()}
        out: list[EnvRollout] = self._all("rollout", [(cpu, version, steps)] * self.num_envs)
        for i, r in enumerate(out):
            if r.version != version or len(r.obs) != steps:
                raise RuntimeError(
                    f"env worker {i} returned version {r.version} with {len(r.obs)} steps; "
                    f"expected version {version} with {steps}"
                )
        return out

    def call(self, name: str, *args: Any) -> tuple[Any, ...]:
        return tuple(self._all("call", [(name, args)] * self.num_envs))

    def get_attr(self, name: str) -> tuple[Any, ...]:
        return tuple(self._all("get", [name] * self.num_envs))

    def set_attr(self, name: str, values: Sequence[Any]) -> None:
        if len(values) != self.num_envs:
            raise ValueError(f"{len(values)} values for {self.num_envs} env workers")
        self._all("set", [(name, v) for v in values])

    def close(self, terminate: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        if not terminate:
            for remote in self._remotes:
                try:
                    remote.send(("close", None))
                    remote.recv()
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
        for proc in self._procs:
            if terminate:
                proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join()
        for remote in self._remotes:
            remote.close()
