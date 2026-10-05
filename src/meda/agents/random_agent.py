"""Uniform over legal candidates (SPEC 11.2). The weakest baseline and an env test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from meda.envs.encoding import Obs


class RandomAgent:
    """Picks a legal candidate uniformly at random. `deterministic` is ignored: the agent is
    reproducible through its seed."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Obs, mask: npt.NDArray[np.bool_], deterministic: bool) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise ValueError("RandomAgent.act: no legal candidate in the mask")
        return int(legal[self.rng.integers(legal.size)])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"agent": self.name, "seed": self.seed}))

    @classmethod
    def load(cls, path: str | Path) -> RandomAgent:
        return cls(seed=int(json.loads(Path(path).read_text())["seed"]))
