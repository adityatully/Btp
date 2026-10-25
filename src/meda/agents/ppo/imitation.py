"""Imitation of a constructive teacher: demonstrations and the cross-entropy loss (SPEC 11.4,
*Imitation warm start*; paper log D-30, D-31).

The teacher is `cheapest_construction` (SPEC 10.3). A demonstration is its plan played through
the env: at every step the observation and the index of the teacher's action in the env's
candidate list. The network learns to put its probability on that index (behaviour cloning);
nothing here changes the env, its reward or the action space, so the same network continues
under PPO afterwards.

The candidate list lists each mix once, with its operands in the env's order, while a
construction may write the same mix the other way round (`Mix(a, k, b, l)` is
`Mix(b, l, a, k)`, SPEC 5.5), so actions are matched up to that swap. Observations are stored
trimmed, as in the rollout buffer, and padded per minibatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from torch import Tensor

from meda.agents.ppo.networks import SetEncoderNet, TensorObs
from meda.agents.ppo.policy import masked_distribution
from meda.agents.ppo.rollout import pad_obs, split_obs
from meda.core.actions import Action, Mix
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.envs.sample_prep_env import SamplePrepEnv
from meda.solvers.fallback import cheapest_construction


def teacher_index(legal: Sequence[Action], action: Action, instance_id: str) -> int:
    """The index of `action` in the candidate list, matching a mix up to operand order."""
    swapped = Mix(action.b, action.l, action.a, action.k) if isinstance(action, Mix) else action
    for i, candidate in enumerate(legal):
        if candidate in (action, swapped):
            return i
    raise ValueError(f"instance '{instance_id}': teacher action {action} is not a candidate")


@dataclass
class Demonstrations:
    """Teacher steps: trimmed observations ([1, ...] each) and the teacher's candidate index."""

    obs: list[dict[str, npt.NDArray[Any]]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    episodes: int = 0

    def __len__(self) -> int:
        return len(self.actions)

    def add_episode(self, env: SamplePrepEnv, instance: Instance, weights: CostWeights) -> None:
        """Play the teacher's plan for `instance` through `env`, recording every step."""
        found = cheapest_construction(instance, weights)
        if found is None:
            raise ValueError(f"instance '{instance.instance_id}': no construction builds it")
        obs, _ = env.reset(options={"instance": instance})
        info: dict[str, Any] = {}
        for action in found[0]:
            index = teacher_index(env.legal, action, instance.instance_id)
            self.obs.extend(split_obs({k: v[None] for k, v in obs.items()}))
            self.actions.append(index)
            obs, _, _, _, info = env.step(index)
        if not info.get("success"):
            raise RuntimeError(
                f"instance '{instance.instance_id}': the teacher's plan ended with status "
                f"{info.get('status')} in the env (step limit {instance.step_limit})"
            )
        self.episodes += 1

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
