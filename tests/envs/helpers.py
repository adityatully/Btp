"""Env construction helpers for tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import gymnasium as gym
import numpy as np

from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.core.state import PrepState
from meda.envs.sample_prep_env import EnvConfig, FixedInstances, SamplePrepEnv
from meda.generators.sampler import dilution_grid
from meda.solvers.baselines.naive import naive_plan
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


def tiny_grid(denominator: int = 16) -> list[Instance]:
    """Every dilution n/denominator at M = 2, each with a step limit twice its naive plan.

    A deterministic, complete stand-in for stage 1 in env tests (the sampler is random).
    """
    out = []
    for inst in dilution_grid(chip("dmfb_m2"), denominator, step_limit=10_000):
        plan = naive_plan(inst)
        assert plan is not None, inst.instance_id
        out.append(replace(inst, step_limit=2 * len(plan)))
    return out


def make_tiny_env() -> SamplePrepEnv:
    """Module-level factory (picklable for AsyncVectorEnv)."""
    return make_env(tiny_grid(), L_pad=128)


class AnyIndexToLegal(gym.ActionWrapper[dict, np.int64, np.int64]):  # type: ignore[type-arg]
    """For gymnasium's check_env only: it steps with unmasked random indices, which our env
    rejects by design. Maps any index onto the legal candidates."""

    def action(self, action: np.int64) -> np.int64:
        base = self.env.unwrapped
        assert isinstance(base, SamplePrepEnv)
        return np.int64(int(action) % len(base.legal))
