"""Masked categorical over candidates, and `PPOAgent` (SPEC 11.1, 11.4).

Masked logits are set to the dtype's minimum, so a masked candidate has probability exactly 0
and can never be sampled (rule 4). Observations are trimmed before the network sees them: every
token axis (fluids, droplets, targets, candidates) is cut to the largest real row count in the
batch, since padding carries no information (SPEC 11.4, batch trimming).

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
from meda.envs.encoding import Obs

# Token axes of a batched observation ([B, ...]): the mask that counts each axis's real rows,
# and, per key, which of its array axes run along it.
AXIS_MASKS = {
    "fluid": "fluid_mask",
    "droplet": "droplet_mask",
    "target": "target_mask",
    "cand": "action_mask",
}
KEY_AXES: dict[str, tuple[tuple[int, str], ...]] = {
    "fluids": ((1, "fluid"),),
    "fluid_mask": ((1, "fluid"),),
    "droplets": ((1, "droplet"),),
    "droplet_mask": ((1, "droplet"),),
    "droplet_comp": ((1, "droplet"), (2, "fluid")),
    "targets": ((1, "target"),),
    "target_mask": ((1, "target"),),
    "target_comp": ((1, "target"), (2, "fluid")),
    "droplet_target": ((1, "droplet"), (2, "target")),
    "candidates": ((1, "cand"),),
    "cand_comp": ((1, "cand"), (2, "fluid")),
    "cand_target": ((1, "cand"), (2, "target")),
    "cand_index": ((1, "cand"),),
    "action_mask": ((1, "cand"),),
    "globals": (),
}
PAD_FILL = {"cand_index": -1}  # padded operands point nowhere; everything else pads with 0


def real_widths(obs: Mapping[str, npt.NDArray[Any]]) -> dict[str, int]:
    """Largest real row count per token axis in a batch (at least 1, so no axis is empty)."""
    return {axis: max(1, int(obs[mask].sum(axis=-1).max())) for axis, mask in AXIS_MASKS.items()}


def trim_obs(
    obs: Mapping[str, npt.NDArray[Any]], widths: Mapping[str, int] | None = None
) -> dict[str, npt.NDArray[Any]]:
    """Cut every token axis of a batched observation to `widths` (default: its real widths)."""
    widths = widths or real_widths(obs)
    out = {}
    for key, value in obs.items():
        slices: list[slice] = [slice(None)] * value.ndim
        for dim, axis in KEY_AXES[key]:
            slices[dim] = slice(0, widths[axis])
        out[key] = value[tuple(slices)]
    return out


def batch_obs(obs: Mapping[str, npt.NDArray[Any]], device: torch.device) -> TensorObs:
    """A batched numpy observation as tensors on `device`, trimmed."""
    return {k: torch.as_tensor(v, device=device) for k, v in trim_obs(obs).items()}


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
        """Weights and the config needed to rebuild the network (not the optimiser). No
        padding size is stored: the network does not depend on one."""
        torch.save(
            {
                "agent": self.name,
                "net": asdict(self.net.cfg),
                "state_dict": self.net.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> PPOAgent:
        data = torch.load(Path(path), map_location=device, weights_only=True)
        net = SetEncoderNet(NetConfig(**data["net"]))
        net.load_state_dict(data["state_dict"])
        net.eval()
        return cls(net, device)
