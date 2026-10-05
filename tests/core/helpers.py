"""Helpers for core tests: instances built from the real chip configs, random rollouts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from meda.core.actions import Action, Output
from meda.core.chip import ChipLimits
from meda.core.composition import ZERO, Composition, Rational
from meda.core.costs import CostWeights, StepCost
from meda.core.dag import DagDelta
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import apply, legal_actions
from meda.core.state import PrepState

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(rel: str) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(CONFIGS / rel))
    assert isinstance(data, dict)
    return data  # type: ignore[return-value]


def chip(name: str = "meda_m8", **overrides: Any) -> ChipLimits:
    return replace(ChipLimits.from_dict(load_yaml(f"chip/{name}.yaml")), **overrides)


def default_weights() -> CostWeights:
    return CostWeights.from_dict(load_yaml("costs/default.yaml"))


SB = (Fluid(0, "S", FluidKind.SAMPLE), Fluid(1, "B", FluidKind.BUFFER))
SRB = (
    Fluid(0, "S", FluidKind.SAMPLE),
    Fluid(1, "R", FluidKind.REAGENT),
    Fluid(2, "B", FluidKind.BUFFER),
)


def dilution(
    sample_shares: list[tuple[int, int]],
    chip_limits: ChipLimits | None = None,
    demand: int = 1,
    droplet_volume: int = 1,
    tolerance: Rational = ZERO,
    step_limit: int = 30,
) -> Instance:
    """Sample S + buffer B; each target is a sample share `n/d`."""
    targets = tuple(
        Target(i, 0, Composition.from_ints((n, d - n), d), tolerance, droplet_volume, demand)
        for i, (n, d) in enumerate(sample_shares)
    )
    return Instance("test-dilution", SB, targets, chip_limits or chip(), step_limit)


def mixture(
    ratios: list[tuple[int, ...]],
    chip_limits: ChipLimits | None = None,
    demand: int = 1,
    tolerance: Rational = ZERO,
    step_limit: int = 30,
) -> Instance:
    targets = tuple(
        Target(i, i, Composition.from_ratio(r), tolerance, 1, demand) for i, r in enumerate(ratios)
    )
    return Instance("test-mixture", SRB, targets, chip_limits or chip(), step_limit)


Transition = tuple[PrepState, Action, PrepState, StepCost, DagDelta]


def random_rollout(instance: Instance, seed: int, prefer_outputs: bool = False) -> list[Transition]:
    """Uniformly random legal actions until terminal.

    With `prefer_outputs`, a random legal Output is taken whenever one exists, so episodes
    finish far more often (used to test complete plans)."""
    rng = np.random.default_rng(seed)
    state = PrepState.initial(instance)
    out: list[Transition] = []
    while True:
        acts = legal_actions(state, instance)
        if not acts:
            return out
        outputs = [a for a in acts if isinstance(a, Output)] if prefer_outputs else []
        pool = outputs or acts
        act = pool[int(rng.integers(len(pool)))]
        nxt, cost, delta = apply(state, act, instance)
        out.append((state, act, nxt, cost, delta))
        state = nxt


# Short step limits keep random rollouts cheap: each M = 8 step enumerates ~1,000 mixes.
ROLLOUT_INSTANCES = {
    "dilution_m8": dilution([(1, 4), (3, 8)], demand=2, step_limit=12),
    "dilution_m2": dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2"), step_limit=12),
    "mixture_m8": mixture([(2, 1, 1), (1, 1, 2)], demand=2, step_limit=12),
    "tight_storage": dilution([(1, 3)], chip_limits=chip(storage_capacity=2), step_limit=12),
    "no_partial": dilution(
        [(1, 2), (1, 4)],
        chip_limits=chip(allow_partial_take=False),
        droplet_volume=2,
        step_limit=12,
    ),
    "merge_on": mixture(
        [(1, 1, 1)], chip_limits=chip(allow_same_composition_merge=True), step_limit=12
    ),
    "tolerance": dilution([(1, 3)], tolerance=Rational.of(1, 20), droplet_volume=2, step_limit=12),
}


# Instances where output-preferring random play usually finishes (complete DAGs).
FINISHING_INSTANCES = {
    "sharing_m2": dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2"), step_limit=40),
    "dilution_m8": dilution(
        [(1, 2), (1, 4), (3, 4)], demand=2, tolerance=Rational.of(1, 16), step_limit=40
    ),
    "mixture_m8": mixture(
        [(1, 1, 0), (0, 1, 1), (1, 1, 1)], tolerance=Rational.of(1, 8), step_limit=40
    ),
    "tolerance": dilution(
        [(1, 3), (2, 3)], tolerance=Rational.of(1, 10), droplet_volume=2, step_limit=60
    ),
}
