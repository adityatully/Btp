"""Masked categorical over candidates, and `PPOAgent` (SPEC 11.1, 11.4).

Masked logits are set to the dtype's minimum, so a masked candidate has probability exactly 0
and can never be sampled (rule 4). Observations are trimmed to the longest real candidate list
in the batch before the network sees them: padding up to `L_max` carries no information.

`PPOAgent` implements the `Agent` protocol, so `run_episode` and the M7 harness grade it like
any other producer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.distributions import Categorical

from meda.agents.ppo.networks import NetConfig, SetEncoderNet, TensorObs
from meda.envs.encoding import EnvSizes, Obs

CANDIDATE_KEYS = ("candidates", "cand_index", "action_mask")


def trim_candidates(obs: Mapping[str, npt.NDArray[Any]]) -> dict[str, npt.NDArray[Any]]:
    """Cut a batched observation ([B, ...]) to the longest real candidate list in the batch."""
    width = max(1, int(obs["action_mask"].sum(axis=-1).max()))
    return {k: (v[:, :width] if k in CANDIDATE_KEYS else v) for k, v in obs.items()}


def batch_obs(obs: Mapping[str, npt.NDArray[Any]], device: torch.device) -> TensorObs:
    """A batched numpy observation as tensors on `device`, trimmed."""
    return {k: torch.as_tensor(v, device=device) for k, v in trim_candidates(obs).items()}


def masked_distribution(logits: Tensor, mask: Tensor) -> Categorical:
    """Categorical over candidates; `mask` [B, L] is 1 for legal ones."""
    fill = torch.finfo(logits.dtype).min
    return Categorical(logits=logits.masked_fill(mask == 0, fill))


class PPOAgent:
    """Greedy (argmax) or sampled actions from a `SetEncoderNet`."""

    name = "ppo"

    def __init__(self, net: SetEncoderNet, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.net = net.to(self.device)

    def act(self, obs: Obs, mask: npt.NDArray[np.bool_], deterministic: bool) -> int:
        batched = {k: v[None] for k, v in obs.items()}
        with torch.no_grad():
            t = batch_obs(batched, self.device)
            logits, _ = self.net(t)
            dist = masked_distribution(logits, t["action_mask"])
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
        index = int(action.item())
        if not mask[index]:
            raise RuntimeError(f"PPOAgent chose masked candidate {index}")  # cannot happen
        return index

    def save(self, path: str | Path) -> None:
        """Weights and the shapes needed to rebuild the network (not the optimiser)."""
        torch.save(
            {
                "agent": self.name,
                "sizes": asdict(self.net.sizes),
                "net": asdict(self.net.cfg),
                "state_dict": self.net.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> PPOAgent:
        data = torch.load(Path(path), map_location=device, weights_only=True)
        net = SetEncoderNet(EnvSizes(**data["sizes"]), NetConfig(**data["net"]))
        net.load_state_dict(data["state_dict"])
        net.eval()
        return cls(net, device)
