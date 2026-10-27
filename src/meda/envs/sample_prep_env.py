"""`SamplePrepEnv`: Gymnasium environment over the exact core (SPEC section 9).

- Action: `Discrete(L_pad)`, an index into `rules.legal_actions(state)` (canonical order).
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
- `snapshot()` / `restore()` save and put back everything the env holds between steps: the
  instance, state, candidates, the plan so far and the env's generator. A run resumed from a
  checkpoint therefore continues its half-finished episodes exactly (SPEC 9.1). Vector envs
  read and write it per worker as the `env_snapshot` attribute (`get_attr` / `set_attr`), then
  `call("observe")` rebuilds each worker's current observation. The vector env must use
  `AutoresetMode.SAME_STEP`: under the default `NEXT_STEP`, an async worker keeps its "reset
  on the next step" flag in a local variable that no snapshot can reach.
"""

from __future__ import annotations

import copy
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
from meda.envs.encoding import Encoder, EnvSizes, HintSource, Obs

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


COST_TO_FINISH = ("share", "construction")
HINT_SOURCES = ("planner", "problem")  # as `solvers.hints.HINT_SOURCES`


@dataclass(frozen=True, slots=True)
class EnvConfig:
    sizes: EnvSizes
    preference_conditioning: bool = False
    # Truncation / dead-end penalty = this x `cost_to_finish` (SPEC 5.8). At 1, giving up costs
    # little more than the construction, so a finished plan worse than that earns less than
    # failing on purpose; 2 keeps finishing preferred up to about twice the construction.
    failure_penalty_factor: int = 1
    # Which completion prices an unfinished plan (SPEC 5.8, 10.3): "share" finishes from the
    # current state reusing the droplets on chip; "construction" wastes them and rebuilds.
    # The env takes the function itself as `cost_to_finish`; this records the choice for
    # whoever builds it (`analysis.evaluate.build_env`).
    cost_to_finish: str = "construction"
    # Where the recipe hint columns come from (SPEC 9.3): "planner" = the tree's and the
    # sharing pass's recipes; "problem" = a deeper pure-fluid chain, no planner (Q-105).
    # Recorded here for whoever builds the hint source (`analysis.evaluate.build_env`).
    hint_source: str = "planner"

    def __post_init__(self) -> None:
        if self.failure_penalty_factor < 1:
            raise ValueError(
                f"failure_penalty_factor must be >= 1, got {self.failure_penalty_factor}"
            )
        if self.cost_to_finish not in COST_TO_FINISH:
            raise ValueError(
                f"cost_to_finish must be one of {COST_TO_FINISH}, got {self.cost_to_finish!r}"
            )
        if self.hint_source not in HINT_SOURCES:
            raise ValueError(f"hint_source must be one of {HINT_SOURCES}, got {self.hint_source!r}")

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> EnvConfig:
        """From `configs/env/sample_prep.yaml`."""
        expected = {
            "F_pad",
            "T_pad",
            "N_pad",
            "L_pad",
            "volume_ref",
            "count_ref",
            "preference_conditioning",
            "failure_penalty_factor",
            "cost_to_finish",
        }
        if set(data) - {"hint_source"} != expected:  # optional: configs before SPEC 0.28
            raise ValueError(f"env config keys must be {sorted(expected)}, got {sorted(data)}")
        return EnvConfig(
            EnvSizes(
                f_pad=int(data["F_pad"]),
                t_pad=int(data["T_pad"]),
                n_pad=int(data["N_pad"]),
                l_pad=int(data["L_pad"]),
                volume_ref=int(data["volume_ref"]),
                count_ref=int(data["count_ref"]),
            ),
            preference_conditioning=bool(data["preference_conditioning"]),
            failure_penalty_factor=int(data["failure_penalty_factor"]),
            cost_to_finish=str(data["cost_to_finish"]),
            hint_source=str(data.get("hint_source", "planner")),
        )


@dataclass(frozen=True, slots=True)
class EnvSnapshot:
    """Everything `SamplePrepEnv` holds between steps (picklable; nothing is shared with the
    env it came from). The instance source and cost-to-finish are not included: they are
    fixed at construction, and the source draws only through `rng`."""

    instance: Instance | None
    state: PrepState | None
    legal: tuple[Action, ...]
    born: tuple[tuple[int, int], ...]
    deltas: tuple[DagDelta, ...]
    totals: StepCost | None
    plan_cost: Rational
    done: bool
    rng: np.random.Generator
    rng_seed: int | None


class SamplePrepEnv(gym.Env[Obs, np.int64]):
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        config: EnvConfig,
        instance_source: InstanceSource,
        cost_weights: CostWeights,
        cost_to_finish: CostToFinish,
        hints: HintSource | None = None,
    ) -> None:
        if config.preference_conditioning:
            raise NotImplementedError("preference conditioning is reserved for milestone 9")
        self.config = config
        self.encoder = Encoder(config.sizes, hints)
        self.observation_space = cast("gym.spaces.Space[Obs]", self.encoder.observation_space())
        self.action_space = gym.spaces.Discrete(config.sizes.l_pad)
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
        mask = np.zeros(self.config.sizes.l_pad, dtype=bool)
        mask[: len(self._legal)] = True
        return mask

    # ------------------------------------------------------------------ snapshot / restore

    def snapshot(self) -> EnvSnapshot:
        """The env's full between-step state, for exact resume from a checkpoint."""
        return EnvSnapshot(
            instance=self._instance,
            state=self._state,
            legal=self._legal,
            born=tuple(sorted(self._born.items())),
            deltas=tuple(self._deltas),
            totals=self._totals,
            plan_cost=self._plan_cost,
            done=self._done,
            rng=copy.deepcopy(self.np_random),
            rng_seed=self._np_random_seed,
        )

    def restore(self, snap: EnvSnapshot) -> None:
        """Put back a `snapshot()`. Raises if its instance does not fit this env's padding."""
        if snap.instance is not None:
            self.encoder.check_instance(snap.instance)
        self._instance = snap.instance
        self._state = snap.state
        self._legal = snap.legal
        self._born = dict(snap.born)
        self._deltas = list(snap.deltas)
        self._totals = snap.totals
        self._plan_cost = snap.plan_cost
        self._done = snap.done
        self._np_random = copy.deepcopy(snap.rng)
        self._np_random_seed = snap.rng_seed

    @property
    def env_snapshot(self) -> EnvSnapshot:
        """`snapshot()` as an attribute, so vector envs can get and set it per worker."""
        return self.snapshot()

    @env_snapshot.setter
    def env_snapshot(self, snap: EnvSnapshot) -> None:
        self.restore(snap)

    def observe(self) -> Obs:
        """The current observation, as the last `reset` or `step` returned it."""
        return self.encoder.encode(self.state, self.instance, self._legal, self._born)

    def _check_penalty_defined(self, instance: Instance) -> None:
        """Refuse, at reset, an instance whose truncation penalty cannot be computed, rather
        than failing mid-episode when the step limit is hit (rule 9). Applies to a
        `cost_to_finish` that exposes `supports(instance)` (the naive fallback, SPEC 10.3)."""
        supports = getattr(self.cost_to_finish, "supports", None)
        if supports is not None and not supports(instance):
            raise ValueError(
                f"instance '{instance.instance_id}': cost_to_finish "
                f"{type(self.cost_to_finish).__name__} cannot price an unfinished plan for it "
                f"(a mixture the naive construction cannot build; SPEC 10.3)"
            )

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        instance = (options or {}).get("instance") or self._source(self.np_random)
        self.encoder.check_instance(instance)
        self._check_penalty_defined(instance)
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
            penalty = self.cost_to_finish(nxt, inst) * self.config.failure_penalty_factor
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
