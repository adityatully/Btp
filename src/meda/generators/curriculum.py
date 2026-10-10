"""Staged difficulty schedule with a promotion rule (SPEC 8.3).

A curriculum is an ordered list of generator stages. The agent is promoted to the next stage
once the last `window` finished episodes reach `min_success_rate` and, when the stage sets a
`reference_cost`, a mean plan cost within `cost_within` of it. Demotion never happens: the
stages are a syllabus, not a controller.

`Curriculum` matches the protocol `envs.wrappers.CurriculumWrapper` expects (`source()` and
`record(info)`), so the env draws each episode from the current stage.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from meda.core.chip import ChipLimits
from meda.core.composition import Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.generators.sampler import GeneratorConfig, sample_instance

# Stage order of the research ladder (SPEC 8.3); the configs live in `configs/generator/`.
LADDER = (
    "tiny_dilution",
    "dilution_single",
    "dilution_multi",
    "mixture_single",
    "mixture_multi",
)


@dataclass(frozen=True, slots=True)
class Stage:
    """One rung: a generator config, its chip, and the rule for leaving it."""

    config: GeneratorConfig
    chip: ChipLimits
    min_success_rate: float = 0.95
    window: int = 200
    reference_cost: Rational | None = None
    cost_within: float = 0.05

    @property
    def name(self) -> str:
        return self.config.name

    def promotes(self, stats: StageStats) -> bool:
        if stats.finished < self.window:
            return False
        if stats.success_rate < self.min_success_rate:
            return False
        if self.reference_cost is None:
            return True
        mean = stats.mean_success_cost
        if mean is None:
            return False
        return mean <= float(self.reference_cost) * (1.0 + self.cost_within)


@dataclass(slots=True)
class StageStats:
    """Rolling window of finished episodes on one stage."""

    window: int
    successes: deque[bool] = field(default_factory=deque)
    costs: deque[float] = field(default_factory=deque)

    def record(self, success: bool, plan_cost: float) -> None:
        self.successes.append(success)
        self.costs.append(plan_cost if success else float("nan"))
        while len(self.successes) > self.window:
            self.successes.popleft()
            self.costs.popleft()

    @property
    def finished(self) -> int:
        return len(self.successes)

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else 0.0

    @property
    def mean_success_cost(self) -> float | None:
        done = [c for c in self.costs if not np.isnan(c)]
        return float(np.mean(done)) if done else None


class Curriculum:
    """Draws instances from the current stage and promotes when its rule is met."""

    def __init__(
        self,
        stages: Sequence[Stage],
        rng: np.random.Generator,
        weights: CostWeights | None = None,
    ) -> None:
        if not stages:
            raise ValueError("a curriculum needs at least one stage")
        self.stages = tuple(stages)
        self.rng = rng
        self.weights = weights
        self.index = 0
        self.stats = StageStats(self.stages[0].window)
        self.promotions: list[dict[str, Any]] = []

    @property
    def stage(self) -> Stage:
        return self.stages[self.index]

    @property
    def finished_all(self) -> bool:
        return self.index == len(self.stages) - 1

    def source(self) -> Any:
        """An instance source for the current stage (`rng -> Instance`)."""
        stage = self.stage

        def draw(rng: np.random.Generator) -> Instance:
            return sample_instance(stage.config, stage.chip, rng, self.weights)

        return draw

    def record(self, episode_info: Mapping[str, Any]) -> None:
        """Take one finished episode's `info` and promote if the rule is now met."""
        if "success" not in episode_info:
            raise KeyError("curriculum.record needs a finished episode's info ('success' missing)")
        cost = episode_info.get("plan_cost")
        self.stats.record(
            bool(episode_info["success"]), float(cost) if cost is not None else float("nan")
        )
        if not self.finished_all and self.stage.promotes(self.stats):
            self.promotions.append(
                {
                    "from": self.stage.name,
                    "to": self.stages[self.index + 1].name,
                    "success_rate": self.stats.success_rate,
                    "mean_cost": self.stats.mean_success_cost,
                }
            )
            self.index += 1
            self.stats = StageStats(self.stage.window)

    def state_dict(self) -> dict[str, Any]:
        """For checkpoints: which stage, and the promotions so far."""
        return {"stage_index": self.index, "promotions": list(self.promotions)}

    def load_state_dict(self, data: Mapping[str, Any]) -> None:
        index = int(data["stage_index"])
        if not 0 <= index < len(self.stages):
            raise ValueError(f"stage_index {index} outside 0..{len(self.stages) - 1}")
        self.index = index
        self.promotions = [dict(p) for p in data.get("promotions", [])]
        self.stats = StageStats(self.stage.window)
