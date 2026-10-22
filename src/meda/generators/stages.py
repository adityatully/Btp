"""Stage design over constraint cells, and the source that draws from cells by weight (SPEC 8.3).

A **cell** fixes one value on each axis (fluids, targets, demand, storage); its generator config
is the design's `base` updated by each value's keys and by every matching override. Stage 1 is
the `start` cell. Each later stage adds one axis value, and its cells are every combination of
the values unlocked so far that no earlier stage holds, minus the held-out cells, which are
never drawn in training.

`MixtureSource` is the env instance source a training run uses: it picks a cell by weight and
draws a fresh problem from that cell's `GeneratorSource`. Which weights is the training
regime's decision (`generators.regime`).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from meda.core.chip import ChipLimits
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.generators.sampler import GeneratorConfig, GeneratorSource

AXES = ("fluids", "targets", "demand", "storage")


@dataclass(frozen=True, slots=True)
class Cell:
    """One constraint combination and the generator config that draws from it."""

    values: tuple[str, ...]  # one per axis, in AXES order
    config: GeneratorConfig

    @property
    def name(self) -> str:
        return "-".join(self.values)


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    adds: tuple[str, str] | None  # (axis, value); None for the start stage
    cells: tuple[Cell, ...]
    val: str  # the frozen validation set of this stage's cells


@dataclass(frozen=True, slots=True)
class StageDesign:
    name: str
    chip: str
    costs: str
    stages: tuple[StageSpec, ...]
    held_out: tuple[Cell, ...]
    held_out_val: str
    final_weights: Mapping[str, float]
    val_size: int
    val_seed: int

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> StageDesign:
        known = {
            "name",
            "chip",
            "costs",
            "base",
            "axes",
            "overrides",
            "start",
            "stages",
            "held_out",
            "final_weights",
            "val_size",
            "val_seed",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"stage design has unknown keys {sorted(unknown)}")
        name = str(data["name"])
        axes = {a: dict(data["axes"][a]) for a in AXES}
        if set(data["axes"]) != set(AXES):
            raise ValueError(f"stage design '{name}': axes must be exactly {AXES}")
        base = dict(data["base"])
        overrides = [(dict(o["when"]), dict(o["set"])) for o in data.get("overrides") or ()]

        def cell(values: Mapping[str, str]) -> Cell:
            for axis, value in values.items():
                if value not in axes[axis]:
                    raise ValueError(f"stage design '{name}': {axis} has no value '{value}'")
            key = tuple(str(values[a]) for a in AXES)
            merged: dict[str, Any] = dict(base)
            for axis, value in zip(AXES, key, strict=True):
                merged.update(axes[axis][value])
            for when, settings in overrides:
                if all(values[a] == v for a, v in when.items()):
                    merged.update(settings)
            merged["name"] = "-".join(key)
            return Cell(key, GeneratorConfig.from_dict(merged))

        held = tuple(cell(dict(c)) for c in data["held_out"]["cells"])
        held_keys = {c.values for c in held}
        unlocked = {a: [str(data["start"][a])] for a in AXES}
        seen: set[tuple[str, ...]] = set()
        stages: list[StageSpec] = []
        for i, entry in enumerate(data["stages"]):
            adds = None
            if i == 0:
                if entry.get("adds"):
                    raise ValueError(f"stage design '{name}': the first stage adds nothing")
            else:
                ((axis, value),) = dict(entry["adds"]).items()
                if value in unlocked[axis]:
                    raise ValueError(f"stage '{entry['name']}' adds {axis}={value} twice")
                unlocked[axis].append(str(value))
                adds = (str(axis), str(value))
            new = []
            for combo in itertools.product(*(unlocked[a] for a in AXES)):
                if combo in seen or combo in held_keys:
                    continue
                seen.add(combo)
                new.append(cell(dict(zip(AXES, combo, strict=True))))
            if not new:
                raise ValueError(f"stage '{entry['name']}' has no cell left to draw")
            stages.append(StageSpec(str(entry["name"]), adds, tuple(new), str(entry["val"])))

        weights = {str(k): float(v) for k, v in dict(data["final_weights"]).items()}
        names = [s.name for s in stages]
        if set(weights) != set(names) or any(w < 0 for w in weights.values()):
            raise ValueError(
                f"stage design '{name}': final_weights needs a non-negative weight for each of "
                f"{names}, got {weights}"
            )
        return StageDesign(
            name=name,
            chip=str(data["chip"]),
            costs=str(data["costs"]),
            stages=tuple(stages),
            held_out=held,
            held_out_val=str(data["held_out"]["val"]),
            final_weights=weights,
            val_size=int(data["val_size"]),
            val_seed=int(data["val_seed"]),
        )

    def first(self, count: int | None) -> StageDesign:
        """The design cut to its first `count` stages (a pilot), final weights renormalised."""
        if count is None or count == len(self.stages):
            return self
        if not 1 <= count <= len(self.stages):
            raise ValueError(
                f"stage design '{self.name}' has {len(self.stages)} stages, not {count}"
            )
        kept = self.stages[:count]
        return replace(self, stages=kept, final_weights=_normalised(self.final_weights, kept))

    def stage(self, name: str) -> StageSpec:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(f"stage design '{self.name}' has no stage '{name}'")

    @property
    def cells(self) -> tuple[Cell, ...]:
        return tuple(c for s in self.stages for c in s.cells)

    def cell_weights(self, stage_weights: Mapping[str, float]) -> dict[str, float]:
        """Each stage's share split evenly over its cells (stages missing get nothing)."""
        out: dict[str, float] = {}
        for s in self.stages:
            share = stage_weights.get(s.name, 0.0)
            for c in s.cells:
                out[c.name] = share / len(s.cells)
        return out

    def source(
        self,
        stage_weights: Mapping[str, float],
        chip: ChipLimits,
        weights: CostWeights | None,
        exclude: frozenset[str],
    ) -> MixtureSource:
        """The training source for these stage shares."""
        by_cell = self.cell_weights(stage_weights)
        parts = tuple(
            (GeneratorSource(c.config, chip, weights, exclude), by_cell[c.name])
            for c in self.cells
            if by_cell[c.name] > 0
        )
        return MixtureSource(parts)


def _normalised(weights: Mapping[str, float], stages: Sequence[StageSpec]) -> dict[str, float]:
    total = sum(weights[s.name] for s in stages)
    if total <= 0:
        raise ValueError(f"final_weights give the stages {[s.name for s in stages]} no weight")
    return {s.name: weights[s.name] / total for s in stages}


@dataclass(frozen=True)
class MixtureSource:
    """An env instance source over several cells: each episode picks a cell with probability
    proportional to its weight, then draws a fresh problem from it (recorded as
    `metadata["cell"]`). Picklable, so it works with `AsyncVectorEnv`."""

    parts: tuple[tuple[GeneratorSource, float], ...]

    def __post_init__(self) -> None:
        if not self.parts or any(w <= 0 for _, w in self.parts):
            raise ValueError("a mixture source needs at least one cell, each with weight > 0")

    @property
    def weights(self) -> dict[str, float]:
        total = sum(w for _, w in self.parts)
        return {src.config.name: w / total for src, w in self.parts}

    def __call__(self, rng: np.random.Generator) -> Instance:
        probs = np.array([w for _, w in self.parts], dtype=np.float64)
        source = self.parts[int(rng.choice(len(self.parts), p=probs / probs.sum()))][0]
        inst = source(rng)
        return replace(inst, metadata={**inst.metadata, "cell": source.config.name})
