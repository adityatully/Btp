"""Tabular Q-learning for tiny instances (SPEC 11.3).

Keys are `state.canonical_key(include_step)` and `action.canonical_key(state)`, **never the
candidate index**: index i means a different action in a different state, so a table keyed on it
would learn nothing transferable. Keying on canonical keys also means two states that differ
only by droplet relabelling share their entries.

`include_step` defaults to **true**, and it matters: these instances have tight step limits (4
for the 1/4 case at M = 4), so the same droplets with the same demand are worth different
amounts at step 1 and at step 3, where there is no longer room to finish. Sharing one entry
between them blurs the two and the greedy policy settles for a worse plan. SPEC 5.4 provides the
flag for exactly this; set it false only when the step limit cannot bind.

Because the keys come from the exact state, this agent cannot work from the observation alone —
the padded float observation does not identify a state. It reads `env.state` and `env.legal`
instead, through `bind(env)`, and that is the honest reason tabular methods stop here: the next
rung of the ladder needs function approximation (SPEC 5.3 of PROJECT_CONTEXT).

**The step size** is a constant by default, not 1/N(s, a). SPEC 11.3 allows either, but 1/N
averages every target a pair ever saw, and with bootstrapping those targets drift upward as the
future is learned, so the early pessimistic ones keep their weight. Measured on the 1/4 case at
M = 4: with 1/N, Q(1:1) settles at -16.02 after 1248 visits where the truth is -15.0, and the
agent keeps the worse 1-mix plan. A constant alpha tracks the moving target.

**The prior for unvisited entries** decides whether this works at all, and `prior` selects it:

- `naive` (default): `-cost_to_finish(state)` from the M5 fallback, the cost of really finishing
  from here with the naive construction. Since naive costs at least the optimum, this is a
  *lower* bound on the achievable return: pessimistic but realistic, so the greedy policy
  prefers what it has learned and exploration is left to epsilon. The same estimate the
  environment charges on truncation also serves as the value prior.
- `fluid_bound`: `-h(state)` with the admissible bound from the exact search (SPEC 10.2), an
  *upper* bound on the return. Optimistic, which drives exploration, but the bound is loose
  (about 8.8 against a true 36.5 for the 7/8 instance), so every unvisited action outranks
  everything learned and the agent keeps chasing optimism instead of settling.
- `zero`: textbook optimistic initialisation. Worst of the three here, because every real return
  is negative, so it is the loosest possible optimism.

This is the "V* is the perfect heuristic" relationship from the research notes, used as a prior
rather than a learned value.

Truncation and dead ends are treated as terminal with no bootstrap. The environment already adds
`cost_to_finish` to that final reward, so bootstrapping a value there would count the rest of
the plan twice.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from meda.core.actions import Action
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.core.state import CanonicalKey
from meda.envs.encoding import Obs
from meda.envs.sample_prep_env import SamplePrepEnv
from meda.solvers.exact_search import fluid_bound_heuristic
from meda.solvers.fallback import NaiveCostToFinish

QKey = tuple[CanonicalKey, CanonicalKey]
PRIORS = ("naive", "fluid_bound", "zero")


@dataclass(frozen=True, slots=True)
class TabularQConfig:
    episodes: int = 8_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_episodes: int = 10_000
    step_size: float | str = 0.5  # a constant alpha, or "visit_count" for 1/N(s, a)
    gamma: float = 1.0
    include_step: bool = True
    prior: str = "naive"  # "naive" | "fluid_bound" | "zero"
    seed: int = 0

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> TabularQConfig:
        known = {f for f in TabularQConfig.__dataclass_fields__ if f != "seed"} | {"name", "seed"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"tabular_q config has unknown keys {sorted(unknown)}")
        step = data.get("step_size", 0.5)
        if isinstance(step, str) and step != "visit_count":
            raise ValueError(f"step_size must be a number or 'visit_count', got {step!r}")
        prior = str(data.get("prior", "naive"))
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {sorted(PRIORS)}, got {prior!r}")
        return TabularQConfig(
            episodes=int(data.get("episodes", 8_000)),
            epsilon_start=float(data.get("epsilon_start", 1.0)),
            epsilon_end=float(data.get("epsilon_end", 0.1)),
            epsilon_decay_episodes=int(data.get("epsilon_decay_episodes", 10_000)),
            step_size=step if isinstance(step, str) else float(step),
            gamma=float(data.get("gamma", 1.0)),
            include_step=bool(data.get("include_step", True)),
            prior=prior,
            seed=int(data.get("seed", 0)),
        )


@dataclass(slots=True)
class TrainingStats:
    """What the run did, for logging and for the acceptance check."""

    episodes: int = 0
    successes: int = 0
    dead_ends: int = 0
    truncations: int = 0
    best_cost: float = math.inf
    best_episode: int = -1
    table_size: int = 0
    returns: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


class TabularQ:
    """Q-learning over canonical state/action keys. Bind it to an env before acting."""

    name = "tabular_q"

    def __init__(
        self, config: TabularQConfig | None = None, weights: CostWeights | None = None
    ) -> None:
        self.config = config or TabularQConfig()
        self.q: dict[QKey, float] = {}
        self.visits: dict[QKey, int] = defaultdict(int)
        self.rng = np.random.default_rng(self.config.seed)
        self.stats = TrainingStats()
        self._env: SamplePrepEnv | None = None
        self._weights = weights
        self._bound = (
            fluid_bound_heuristic(weights)
            if weights is not None and self.config.prior == "fluid_bound"
            else None
        )
        self._fallback = (
            NaiveCostToFinish(weights)
            if weights is not None and self.config.prior == "naive"
            else None
        )

    # ------------------------------------------------------------------ binding

    def bind(self, env: SamplePrepEnv) -> None:
        """Attach the env this agent reads its state from."""
        self._env = env

    @property
    def env(self) -> SamplePrepEnv:
        if self._env is None:
            raise RuntimeError(
                "TabularQ needs bind(env) before acting: it keys on the exact state, which the "
                "padded observation does not identify"
            )
        return self._env

    # ------------------------------------------------------------------ values

    def state_key(self, state: Any) -> CanonicalKey:
        return state.canonical_key(self.config.include_step)

    def initial_value(self, state: Any, instance: Instance) -> float:
        """The value of an entry with no data yet (see the module docstring)."""
        if self._fallback is not None:
            return -float(self._fallback(state, instance))
        if self._bound is not None:
            return -float(self._bound(state, instance))
        return 0.0

    def value(self, state_key: CanonicalKey, action_key: CanonicalKey, default: float) -> float:
        return self.q.get((state_key, action_key), default)

    def greedy_index(
        self,
        legal: Sequence[Action],
        state: Any,
        instance: Instance,
        prefer_visited: bool = False,
    ) -> int:
        """Index of the best legal candidate; ties go to the first in canonical order.

        `prefer_visited` makes an action the agent has data for beat one it does not. That is
        only wanted with an *optimistic* prior, where following the prior means chasing a guess;
        with the pessimistic `naive` prior an unvisited entry already looks no better than
        finishing naively, so the rule would only throw away real values -- and it costs the
        optimum wherever two plans are close (the 1/4 case at M = 4 is decided by 0.5).
        """
        state_key = self.state_key(state)
        default = self.initial_value(state, instance)
        best_index, best_value, best_seen = 0, -math.inf, False
        for i, action in enumerate(legal):
            key = (state_key, action.canonical_key(state))
            seen = key in self.q
            if prefer_visited and best_seen and not seen:
                continue
            v = self.q.get(key, default)
            if (prefer_visited and seen and not best_seen) or v > best_value:
                best_index, best_value, best_seen = i, v, seen
        return best_index

    def best_value(self, state: Any, legal: Sequence[Action], instance: Instance) -> float:
        if not legal:
            return 0.0
        key = self.state_key(state)
        default = self.initial_value(state, instance)
        return max(self.value(key, a.canonical_key(state), default) for a in legal)

    # ------------------------------------------------------------------ Agent protocol

    def act(self, obs: Obs, mask: npt.NDArray[np.bool_], deterministic: bool) -> int:
        env = self.env
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(env.legal)))
        return self.greedy_index(env.legal, env.state, env.instance, prefer_visited=deterministic)

    @property
    def epsilon(self) -> float:
        cfg = self.config
        if cfg.epsilon_decay_episodes <= 0:
            return cfg.epsilon_end
        fraction = min(1.0, self.stats.episodes / cfg.epsilon_decay_episodes)
        return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)

    def alpha(self, key: QKey) -> float:
        if self.config.step_size == "visit_count":
            return 1.0 / self.visits[key]
        assert isinstance(self.config.step_size, float)
        return self.config.step_size

    # ------------------------------------------------------------------ training

    def train(
        self, env: SamplePrepEnv, instances: Sequence[Instance], log_every: int = 0
    ) -> TrainingStats:
        """Run `config.episodes` episodes, cycling through `instances` in order.

        Cycling rather than sampling keeps every instance equally represented, which matters
        when the acceptance check is per instance.
        """
        if not instances:
            raise ValueError("tabular Q needs at least one instance to train on")
        self.bind(env)
        cfg = self.config
        for episode in range(cfg.episodes):
            instance = instances[episode % len(instances)]
            env.reset(options={"instance": instance})
            episode_return = 0.0
            while True:
                state, legal = env.state, env.legal
                state_key = self.state_key(state)
                epsilon = self.epsilon
                if self.rng.random() < epsilon:
                    index = int(self.rng.integers(len(legal)))
                else:
                    index = self.greedy_index(legal, state, instance)
                action_key = legal[index].canonical_key(state)
                key = (state_key, action_key)
                default = self.initial_value(state, instance)

                _, reward, terminated, truncated, info = env.step(np.int64(index))
                episode_return += reward
                # Truncation and dead ends are terminal: the reward already includes
                # cost_to_finish, so there is nothing left to bootstrap.
                future = (
                    0.0
                    if terminated or truncated
                    else self.best_value(env.state, env.legal, instance)
                )
                self.visits[key] += 1
                target = reward + cfg.gamma * future
                current = self.q.get(key, default)
                self.q[key] = current + self.alpha(key) * (target - current)
                if terminated or truncated:
                    self._record(info, episode, episode_return)
                    break
            if log_every and (episode + 1) % log_every == 0:
                print(
                    f"  episode {episode + 1:>7d}  eps={epsilon:.3f}  "
                    f"success={self.stats.success_rate:6.1%}  "
                    f"best_cost={self.stats.best_cost:.2f}  |Q|={len(self.q)}"
                )
        self.stats.table_size = len(self.q)
        return self.stats

    def _record(self, info: Mapping[str, Any], episode: int, episode_return: float) -> None:
        stats = self.stats
        stats.episodes += 1
        stats.returns.append(episode_return)
        if info["success"]:
            stats.successes += 1
            cost = float(info["plan_cost"])
            if cost < stats.best_cost:
                stats.best_cost, stats.best_episode = cost, episode
        elif info["dead_end"]:
            stats.dead_ends += 1
        else:
            stats.truncations += 1

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        """Pickle: the keys are tuples of `Composition` objects, which JSON cannot hold."""
        path = Path(path)
        path.write_bytes(
            pickle.dumps(
                {
                    "config": self.config,
                    "weights": self._weights,
                    "q": self.q,
                    "visits": dict(self.visits),
                    "episodes": self.stats.episodes,
                }
            )
        )
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "agent": self.name,
                    "entries": len(self.q),
                    "episodes": self.stats.episodes,
                    "success_rate": self.stats.success_rate,
                    "best_cost": self.stats.best_cost,
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> TabularQ:
        data = pickle.loads(Path(path).read_bytes())
        agent = cls(data["config"], data.get("weights"))
        agent.q = data["q"]
        agent.visits = defaultdict(int, data["visits"])
        agent.stats.episodes = int(data.get("episodes", 0))
        agent.stats.table_size = len(agent.q)
        return agent
