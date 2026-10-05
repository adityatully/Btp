"""`NoiseModel` interface and the default `NoNoise` (SPEC 5.9, phase 2 hook).

Phase 1 is deterministic and `rules.apply` stays pure. In phase 2 the environment will pass the
result of each Mix through a `NoiseModel` (e.g. split error on taken volumes). When any model
other than `NoNoise` is active, the env reports `info["stochastic"] = True` and exact search is
disabled.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from meda.core.actions import Source
from meda.core.state import Droplet


class NoiseModel(Protocol):
    stochastic: bool

    def apply(
        self,
        rng: np.random.Generator,
        mix_result: Droplet,
        a: Source,
        k: int,
        b: Source,
        l: int,
    ) -> Droplet: ...


class NoNoise:
    """Exact transitions: returns the mix result unchanged."""

    stochastic = False

    def apply(
        self,
        rng: np.random.Generator,
        mix_result: Droplet,
        a: Source,
        k: int,
        b: Source,
        l: int,
    ) -> Droplet:
        return mix_result
