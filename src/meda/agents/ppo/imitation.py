"""Imitation of a constructive teacher: demonstrations and the cross-entropy loss (SPEC 11.4,
*Imitation warm start*; paper log D-30, D-31).

The teacher is a plan producer: the sharing pass (`share_plan`, SPEC 10.4; never dearer than
`cheapest_construction`, at the proved optimum on 51 of 65 small problems, E-17) by default, or
`cheapest_construction` (SPEC 10.3). A demonstration is its plan played through the env: at
every step the observation and the index of the teacher's action in the env's candidate list.
The network learns to put its probability on that index (behaviour cloning); nothing here
changes the env, its reward or the action space, so the same network continues under PPO
afterwards.

The candidate list lists each mix once, with its operands in the env's order, while a
construction may write the same mix the other way round (`Mix(a, k, b, l)` is
`Mix(b, l, a, k)`, SPEC 5.5), so actions are matched up to that swap. The list also offers one
move per class of identical droplets (the smallest uid stands for its twins), while the
teacher's plan names a particular droplet; when it names a twin the env did not list, the move
is matched by canonical key and the two droplets swap roles in a uid map, so the teacher's
later actions follow the droplet the env actually used (paper log F-16). Observations are
stored trimmed, as in the rollout buffer, and padded per minibatch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from torch import Tensor

from meda.agents.ppo.networks import SetEncoderNet, TensorObs
from meda.agents.ppo.policy import masked_distribution
from meda.agents.ppo.rollout import pad_obs, split_obs
from meda.core.actions import Action, Discard, Mix, Output, Source
from meda.core.composition import Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.core.state import PrepState
from meda.envs.sample_prep_env import SamplePrepEnv
from meda.solvers.baselines.share import share_plan
from meda.solvers.fallback import cheapest_construction

Teacher = Callable[[Instance, CostWeights], tuple[Action, ...] | None]


def construction_teacher(instance: Instance, weights: CostWeights) -> tuple[Action, ...] | None:
    found = cheapest_construction(instance, weights)
    return None if found is None else found[0]


TEACHERS: dict[str, Teacher] = {"share": share_plan, "construction": construction_teacher}


def teacher_index(
    legal: Sequence[Action], action: Action, instance_id: str, state: PrepState | None = None
) -> int:
    """The index of `action` in the candidate list, matching a mix up to operand order; with
    `state`, an action on a twin the list does not offer matches the move with its canonical
    key."""
    swapped = Mix(action.b, action.l, action.a, action.k) if isinstance(action, Mix) else action
    for i, candidate in enumerate(legal):
        if candidate in (action, swapped):
            return i
    if state is not None:
        key = action.canonical_key(state)
        for i, candidate in enumerate(legal):
            if candidate.canonical_key(state) == key:
                return i
    raise ValueError(f"instance '{instance_id}': teacher action {action} is not a candidate")


@dataclass
class UidMap:
    """Which env droplet each droplet of the teacher's plan is. The plan and the env make the
    same droplets with the same uids, except that where the env acted on a twin, the two
    twins trade places. Identity until then."""

    env_of: dict[int, int] = field(default_factory=dict)

    def source(self, src: Source) -> Source:
        return src if src.is_reservoir else Source.droplet(self.env_of.get(src.index, src.index))

    def action(self, action: Action) -> Action:
        """The teacher's action in env uids."""
        if isinstance(action, Mix):
            return Mix(self.source(action.a), action.k, self.source(action.b), action.l)
        if isinstance(action, Output):
            return Output(self.source(action.src), action.target)
        return Discard(self.env_of.get(action.uid, action.uid))

    def follow(self, meant: Action, played: Action, state: PrepState) -> None:
        """After the env plays `played` for the teacher's `meant` (both in env uids, same
        canonical key in `state`), let each teacher droplet follow the twin the env used.

        The meant -> played uids are closed into a permutation of the droplets involved: each
        chain m -> p -> ... ends at a twin the teacher did not name, which takes the place of
        the chain's first droplet. Every step of a chain stays within one class of twins."""
        sigma = dict(_paired_uids(meant, played, state))
        if all(u == v for u, v in sigma.items()):
            return
        played_uids = set(sigma.values())
        for start in [u for u in sigma if u not in played_uids]:
            end = start
            while end in sigma:
                end = sigma[end]
            sigma[end] = start
        teacher = {e: self._teacher_of(e) for e in sigma}
        for e, t in teacher.items():
            self.env_of[t] = sigma[e]

    def _teacher_of(self, env_uid: int) -> int:
        for t, e in self.env_of.items():
            if e == env_uid:
                return t
        return env_uid


def _paired_uids(meant: Action, played: Action, state: PrepState) -> list[tuple[int, int]]:
    """(meant uid, played uid) for each droplet operand, operands aligned by canonical key."""
    if isinstance(meant, Discard) and isinstance(played, Discard):
        return [(meant.uid, played.uid)]
    if isinstance(meant, Output) and isinstance(played, Output):
        pairs = [(meant.src, played.src)]
    elif isinstance(meant, Mix) and isinstance(played, Mix):
        same = (meant.a.canonical_key(state), meant.k) == (played.a.canonical_key(state), played.k)
        pairs = [(meant.a, played.a), (meant.b, played.b)]
        if not same:
            pairs = [(meant.a, played.b), (meant.b, played.a)]
    else:
        raise ValueError(f"matched actions of different kinds: {meant} and {played}")
    return [(m.index, p.index) for m, p in pairs if not m.is_reservoir]


@dataclass
class Demonstrations:
    """Teacher steps: trimmed observations ([1, ...] each) and the teacher's candidate index."""

    obs: list[dict[str, npt.NDArray[Any]]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    episodes: int = 0

    def __len__(self) -> int:
        return len(self.actions)

    def add_episode(
        self,
        env: SamplePrepEnv,
        instance: Instance,
        weights: CostWeights,
        teacher: Teacher = share_plan,
    ) -> Rational:
        """Play the teacher's plan for `instance` through `env`, recording every step; returns
        the plan's exact cost."""
        plan = teacher(instance, weights)
        if plan is None:
            raise ValueError(f"instance '{instance.instance_id}': the teacher found no plan")
        obs, _ = env.reset(options={"instance": instance})
        info: dict[str, Any] = {}
        uids = UidMap()
        for action in plan:
            meant = uids.action(action)
            index = teacher_index(env.legal, meant, instance.instance_id, env.state)
            uids.follow(meant, env.legal[index], env.state)
            self.obs.extend(split_obs({k: v[None] for k, v in obs.items()}))
            self.actions.append(index)
            obs, _, _, _, info = env.step(index)
        if not info.get("success"):
            raise RuntimeError(
                f"instance '{instance.instance_id}': the teacher's plan ended with status "
                f"{info.get('status')} in the env (step limit {instance.step_limit})"
            )
        self.episodes += 1
        cost: Rational = info["plan_cost"]
        return cost

    def minibatch(self, index: npt.NDArray[np.int64]) -> tuple[dict[str, Any], npt.NDArray[Any]]:
        """Padded observations and teacher indices at `index`."""
        actions = np.array([self.actions[int(i)] for i in index], dtype=np.int64)
        return pad_obs([self.obs[int(i)] for i in index]), actions

    def candidate_widths(self) -> npt.NDArray[np.int64]:
        return np.array([o["action_mask"].shape[1] for o in self.obs], dtype=np.int64)


def imitation_loss(net: SetEncoderNet, obs: TensorObs, actions: Tensor) -> tuple[Tensor, Tensor]:
    """Mean cross-entropy of the teacher's index under the masked policy, and the fraction of
    steps whose most likely candidate is the teacher's."""
    logits, _ = net(obs)
    dist = masked_distribution(logits, obs["action_mask"])
    loss = -dist.log_prob(actions).mean()
    agree = (dist.probs.argmax(-1) == actions).float().mean()
    return loss, agree
