"""Training regimes: which stages a run draws from, and when that changes (SPEC 8.3).

All three regimes share the stage design, the mixture source and the validation sets; they
differ only in the stage shares they ask for:

- `sequential` (E): only the newest stage; a promotion replaces it with the next one.
- `expanding` (C): a promotion adds the next stage. The newest stage gets `newest_share`, the
  earlier ones split the rest evenly; once the last stage passes, the design's final weights.
- `mixed` (D): the design's final weights from the start; nothing is ever promoted.

A stage **passes** at a validation round when its greedy success rate is at least
`min_success` and its mean gap to the best known plan has plateaued: the best gap of the last
`patience` rounds is not `min_improvement` below the best gap before them. A stage that has not
passed after `max_stage_fraction` of the run's env steps is promoted anyway and the event is
marked `forced` (rule 9: never silent). Promotion decisions read only validation results,
never training episodes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from meda.generators.stages import StageDesign

REGIMES = ("sequential", "expanding", "mixed")


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """`configs/regime/*.yaml`. Unknown or missing keys raise."""

    name: str
    num_stages: int | None  # the design's first k stages (a pilot); None = all
    min_success: float
    patience: int
    min_improvement: float
    max_stage_fraction: float
    newest_share: float

    def __post_init__(self) -> None:
        if self.name not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.name!r}")
        if not 0 < self.min_success <= 1 or self.patience < 1 or self.min_improvement < 0:
            raise ValueError(f"regime '{self.name}': bad promotion rule {self}")
        if not 0 < self.max_stage_fraction <= 1 or not 0 < self.newest_share <= 1:
            raise ValueError(f"regime '{self.name}': fractions must lie in (0, 1]")

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> RegimeConfig:
        names = {f.name for f in fields(RegimeConfig)}
        if set(data) != names:
            raise ValueError(
                f"regime config keys mismatch: missing {sorted(names - set(data))}, "
                f"unknown {sorted(set(data) - names)}"
            )
        k = data["num_stages"]
        return RegimeConfig(
            name=str(data["name"]),
            num_stages=None if k is None else int(k),
            min_success=float(data["min_success"]),
            patience=int(data["patience"]),
            min_improvement=float(data["min_improvement"]),
            max_stage_fraction=float(data["max_stage_fraction"]),
            newest_share=float(data["newest_share"]),
        )


class Regime:
    """The stage shares of one training run, updated from validation rounds."""

    def __init__(self, cfg: RegimeConfig, design: StageDesign, total_env_steps: int) -> None:
        self.cfg = cfg
        self.design = design.first(cfg.num_stages)
        self.total_env_steps = total_env_steps
        n = len(self.design.stages)
        if cfg.name != "mixed" and n * cfg.max_stage_fraction > 1 + 1e-9:
            # Forced promotions at every cap would start the last stage at (n - 1) x the
            # fraction of the budget, leaving it less than one cap, or none (5 x 0.3).
            raise ValueError(
                f"regime '{cfg.name}': {n} stages x max_stage_fraction "
                f"{cfg.max_stage_fraction} > 1, so forced promotions could leave the last "
                f"stage(s) untrained; set max_stage_fraction <= {1 / n:.3g}"
            )
        self.newest = 0  # index of the newest stage drawn from (sequential / expanding)
        self.finished = cfg.name == "mixed"  # every stage in, final shares
        self.stage_start = 0  # env step at which the newest stage came in
        self.gaps: list[float] = []  # newest stage's mean gap per round since it came in
        self.events: list[dict[str, Any]] = []

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.design.stages]

    def stage_weights(self) -> dict[str, float]:
        names = self.names
        if self.finished and self.cfg.name != "sequential":
            return dict(self.design.final_weights)
        if self.cfg.name == "sequential" or self.newest == 0:
            return {names[self.newest]: 1.0}
        earlier = (1.0 - self.cfg.newest_share) / self.newest
        return {
            **dict.fromkeys(names[: self.newest], earlier),
            names[self.newest]: self.cfg.newest_share,
        }

    def observe(self, env_step: int, results: Mapping[str, tuple[float, float]]) -> bool:
        """Take one validation round, `{stage: (success_rate, mean gap to best known)}` (the
        gap over successful plans; NaN if none). Return True if the stage shares changed."""
        if self.finished:
            return False
        name = self.names[self.newest]
        success, gap = results[name]
        self.gaps.append(gap)
        passed = success >= self.cfg.min_success and self._plateaued()
        forced = (
            not passed
            and env_step - self.stage_start >= self.cfg.max_stage_fraction * self.total_env_steps
        )
        if not (passed or forced):
            return False
        last = self.newest == len(self.names) - 1
        self.events.append(
            {
                "env_step": env_step,
                "stage": name,
                "event": "finished" if last else "promoted",
                "forced": forced,
                "success_rate": success,
                "gap": None if math.isnan(gap) else gap,
            }
        )
        if last:
            self.finished = True
            return self.cfg.name == "expanding"  # sequential keeps drawing the last stage
        self.newest += 1
        self.stage_start = env_step
        self.gaps = []
        return True

    def _plateaued(self) -> bool:
        p = self.cfg.patience
        if len(self.gaps) <= p or any(math.isnan(g) for g in self.gaps[-p:]):
            return False
        before = [g for g in self.gaps[:-p] if not math.isnan(g)]
        if not before:
            return False
        return min(self.gaps[-p:]) > min(before) - self.cfg.min_improvement

    def state_dict(self) -> dict[str, Any]:
        return {
            "regime": self.cfg.name,
            "stages": self.names,
            "newest": self.newest,
            "finished": self.finished,
            "stage_start": self.stage_start,
            "gaps": list(self.gaps),
            "events": [dict(e) for e in self.events],
        }

    def load_state_dict(self, data: Mapping[str, Any]) -> None:
        if data["regime"] != self.cfg.name or list(data["stages"]) != self.names:
            raise ValueError(
                f"checkpoint regime {data['regime']} over {data['stages']} does not match "
                f"this run's {self.cfg.name} over {self.names}"
            )
        self.newest = int(data["newest"])
        self.finished = bool(data["finished"])
        self.stage_start = int(data["stage_start"])
        self.gaps = [float(g) for g in data["gaps"]]
        self.events = [dict(e) for e in data["events"]]
