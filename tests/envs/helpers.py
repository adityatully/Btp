"""Env construction helpers for tests."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.core.state import PrepState
from meda.envs.sample_prep_env import EnvConfig, FixedInstances, SamplePrepEnv
from meda.generators.sampler import dilution_grid
from tests.core.helpers import chip, default_weights, load_yaml


class ConstantPenalty:
    """Truncation penalty stand-in (the real one is `solvers.fallback`, M5). Picklable."""

    def __init__(self, value: int = 0) -> None:
        self.value = Rational.of(value)

    def __call__(self, state: PrepState, instance: Instance) -> Rational:
        return self.value


def env_config(**overrides: object) -> EnvConfig:
    data = {**load_yaml("env/sample_prep.yaml"), **overrides}
    return EnvConfig.from_dict(data)


def make_env(
    instances: Sequence[Instance], penalty: int = 0, **env_overrides: object
) -> SamplePrepEnv:
    return SamplePrepEnv(
        env_config(**env_overrides),
        FixedInstances(instances),
        default_weights(),
        ConstantPenalty(penalty),
    )


def tiny_grid() -> list[Instance]:
    """Stand-in for the tiny_dilution stage until the M4 sampler: all n/16, M = 2."""
    gen = load_yaml("generator/tiny_dilution.yaml")
    return dilution_grid(chip("dmfb_m2"), gen["accuracy_denominator"], gen["step_limit"])


def make_tiny_env() -> SamplePrepEnv:
    """Module-level factory (picklable for AsyncVectorEnv)."""
    return make_env(tiny_grid(), L_max=128)


class AnyIndexToLegal(gym.ActionWrapper[dict, np.int64, np.int64]):  # type: ignore[type-arg]
    """For gymnasium's check_env only: it steps with unmasked random indices, which our env
    rejects by design. Maps any index onto the legal candidates."""

    def action(self, action: np.int64) -> np.int64:
        base = self.env.unwrapped
        assert isinstance(base, SamplePrepEnv)
        return np.int64(int(action) % len(base.legal))
