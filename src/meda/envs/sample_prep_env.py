"""`SamplePrepEnv`: Gymnasium environment over the exact core (SPEC section 9).

- Action: `Discrete(L_max)`, an index into `rules.legal_actions(state)` (canonical order).
  Selecting an index that is not a legal candidate raises `IllegalActionError`.
- Reward: `-float(step cost)` from exact rational costs (SPEC 6.4).
- Episode end (SPEC 5.8):
  * success -> `terminated=True`;
  * dead end (no legal action) -> `terminated=True`, plus the truncation penalty;
  * step limit -> `truncated=True`, plus the truncation penalty.
  The penalty `cost_to_finish(state, instance)` already estimates the remaining cost, so agents
  must not bootstrap a value after a truncated step.
- `info` every step: `instance_id`, `cost_breakdown`, `legal_count`, `stochastic`. At episode
  end also: `status`, `success`, `dead_end`, `episode_totals`, `plan_cost` (exact, without
  penalty), `penalty` (exact), and `dag` (the episode's `MixingDAG`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from meda.core.actions import Action
from meda.core.composition import ZERO, Rational
from meda.core.costs import CostWeights, StepCost
from meda.core.dag import DagDelta, MixingDAG, NodeType
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError, TerminalStatus, apply, legal_actions
from meda.core.state import PrepState
from meda.envs.encoding import Encoder, EnvSizes, Obs

CostToFinish = Callable[[PrepState, Instance], Rational]


class InstanceSource(Protocol):
    """Start-state distribution p(s0): draws an instance using the env's generator."""

    def __call__(self, rng: np.random.Generator) -> Instance: ...


class FixedInstances:
    """Draws uniformly from a fixed list (tests, profiling, small training sets)."""

    def __init__(self, instances: Sequence[Instance]) -> None:
        if not instances:
            raise ValueError("FixedInstances needs at least one instance")
        self.instances = tuple(instances)

    def __call__(self, rng: np.random.Generator) -> Instance:
        return self.instances[int(rng.integers(len(self.instances)))]


@dataclass(frozen=True, slots=True)
class EnvConfig:
    sizes: EnvSizes
    preference_conditioning: bool = False

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> EnvConfig:
        """From `configs/env/sample_prep.yaml`."""
        expected = {"F_max", "T_max", "storage_capacity_max", "L_max", "preference_conditioning"}
        if set(data) != expected:
            raise ValueError(f"env config keys must be {sorted(expected)}, got {sorted(data)}")
        return EnvConfig(
            EnvSizes(
                f_max=int(data["F_max"]),
                t_max=int(data["T_max"]),
                storage_max=int(data["storage_capacity_max"]),
                l_max=int(data["L_max"]),
            ),
            preference_conditioning=bool(data["preference_conditioning"]),
        )


class SamplePrepEnv(gym.Env[Obs, np.int64]):
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        config: EnvConfig,
        instance_source: InstanceSource,
        cost_weights: CostWeights,
        cost_to_finish: CostToFinish,
    ) -> None:
        if config.preference_conditioning:
            raise NotImplementedError("preference conditioning is reserved for milestone 9")
        self.config = config
        self.encoder = Encoder(config.sizes)
        self.observation_space = cast("gym.spaces.Space[Obs]", self.encoder.observation_space())
        self.action_space = gym.spaces.Discrete(config.sizes.l_max)
        self.cost_weights = cost_weights
        self.cost_to_finish = cost_to_finish
        self._source = instance_source
        self._instance: Instance | None = None
        self._state: PrepState | None = None
        self._legal: tuple[Action, ...] = ()
        self._born: dict[int, int] = {}
        self._deltas: list[DagDelta] = []
        self._totals: StepCost | None = None
        self._plan_cost: Rational = ZERO
        self._done = True

    # ------------------------------------------------------------------ read-only views

    @property
    def state(self) -> PrepState:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    @property
    def instance(self) -> Instance:
        if self._instance is None:
            raise RuntimeError("call reset() first")
        return self._instance

    @property
    def legal(self) -> tuple[Action, ...]:
        """Current candidate list; index i is action i."""
        return self._legal

    def set_instance_source(self, source: InstanceSource) -> None:
        """Takes effect at the next reset (curriculum)."""
        self._source = source

    def action_masks(self) -> npt.NDArray[np.bool_]:
        mask = np.zeros(self.config.sizes.l_max, dtype=bool)
        mask[: len(self._legal)] = True
        return mask

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        instance = (options or {}).get("instance") or self._source(self.np_random)
        self.encoder.check_instance(instance)
        self._instance = instance
        self._state = PrepState.initial(instance)
        self._legal = legal_actions(self._state, instance)
        if not self._legal:
            raise ValueError(f"instance '{instance.instance_id}': no legal action at the start")
        self._born = {}
        self._deltas = []
        self._totals = StepCost.zero(instance.num_fluids)
        self._plan_cost = ZERO
        self._done = False
        obs = self.encoder.encode(self._state, instance, self._legal, self._born)
        return obs, {
            "instance_id": instance.instance_id,
            "legal_count": len(self._legal),
            "stochastic": False,
        }

    def step(self, action: np.int64 | int) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        if self._done or self._state is None or self._instance is None:
            raise RuntimeError("episode is over (or not started); call reset()")
        inst, state = self._instance, self._state
        idx = int(action)
        if not 0 <= idx < len(self._legal):
            raise IllegalActionError(
                f"instance '{inst.instance_id}' step {state.step}: candidate index {idx} is "
                f"masked ({len(self._legal)} legal candidates)"
            )
        nxt, cost, delta = apply(state, self._legal[idx], inst)
        for node in delta.nodes:
            if node.type is NodeType.MIX:
                self._born[node.id] = nxt.step  # age 0 right after the action that made it
        self._deltas.append(delta)
        assert self._totals is not None
        self._totals = self._totals + cost
        step_cost = self.cost_weights.scalar(cost, inst.fluids)
        self._plan_cost = self._plan_cost + step_cost
        self._state = nxt
        self._legal = legal_actions(nxt, inst)

        status: TerminalStatus | None = None
        if nxt.is_success:
            status = TerminalStatus.SUCCESS
        elif nxt.step >= inst.step_limit:
            status = TerminalStatus.TRUNCATED
        elif not self._legal:
            status = TerminalStatus.DEAD_END
        penalty = ZERO
        if status in (TerminalStatus.TRUNCATED, TerminalStatus.DEAD_END):
            penalty = self.cost_to_finish(nxt, inst)
        reward = -float(step_cost + penalty)

        info: dict[str, Any] = {
            "instance_id": inst.instance_id,
            "cost_breakdown": cost,
            "legal_count": len(self._legal),
            "stochastic": False,
        }
        if status is not None:
            self._done = True
            self._legal = ()
            info.update(
                status=status,
                success=status is TerminalStatus.SUCCESS,
                dead_end=status is TerminalStatus.DEAD_END,
                episode_totals=self._totals,
                plan_cost=self._plan_cost,
                penalty=penalty,
                dag=MixingDAG.from_deltas(inst, self._deltas, self._totals, "env"),
            )
        obs = self.encoder.encode(nxt, inst, self._legal, self._born)
        terminated = status in (TerminalStatus.SUCCESS, TerminalStatus.DEAD_END)
        truncated = status is TerminalStatus.TRUNCATED
        return obs, reward, terminated, truncated, info
