"""`Target`, `Instance` and JSON round-trip (SPEC 5.3).

JSON stores compositions and rationals as `{"num": ..., "den": ...}`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from meda.core.chip import ChipLimits
from meda.core.composition import Composition, Rational
from meda.core.fluids import Fluid, FluidKind


@dataclass(frozen=True, slots=True)
class Target:
    """A requested mixture: `demand` droplets of `droplet_volume` units each.

    A droplet satisfies the target if `linf_distance(droplet, composition) <= tolerance`.
    """

    id: int
    experiment_id: int
    composition: Composition
    tolerance: Rational
    droplet_volume: int
    demand: int


@dataclass(frozen=True, slots=True)
class Instance:
    """One sample-preparation problem. Fluid ids are 0..F-1, target ids 0..T-1, in order."""

    instance_id: str
    fluids: tuple[Fluid, ...]
    targets: tuple[Target, ...]
    chip: ChipLimits
    step_limit: int
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        def fail(msg: str) -> None:
            raise ValueError(f"instance '{self.instance_id}': {msg}")

        if not self.fluids:
            fail("needs at least one fluid")
        if [f.id for f in self.fluids] != list(range(len(self.fluids))):
            fail("fluid ids must be 0..F-1 in order")
        if [t.id for t in self.targets] != list(range(len(self.targets))):
            fail("target ids must be 0..T-1 in order")
        if self.step_limit < 1:
            fail("step_limit must be >= 1")
        for t in self.targets:
            if t.composition.num_fluids != len(self.fluids):
                fail(f"target {t.id} composition has {t.composition.num_fluids} fluids")
            if t.tolerance < 0:
                fail(f"target {t.id} tolerance must be >= 0")
            if t.droplet_volume < 1:
                fail(f"target {t.id} droplet_volume must be >= 1")
            if t.demand < 1:
                fail(f"target {t.id} demand must be >= 1")

    @property
    def num_fluids(self) -> int:
        return len(self.fluids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "fluids": [{"id": f.id, "name": f.name, "kind": f.kind.value} for f in self.fluids],
            "targets": [
                {
                    "id": t.id,
                    "experiment_id": t.experiment_id,
                    "composition": {"num": list(t.composition.num), "den": t.composition.den},
                    "tolerance": {"num": t.tolerance.num, "den": t.tolerance.den},
                    "droplet_volume": t.droplet_volume,
                    "demand": t.demand,
                }
                for t in self.targets
            ],
            "chip": self.chip.to_dict(),
            "step_limit": self.step_limit,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Instance:
        return Instance(
            instance_id=data["instance_id"],
            fluids=tuple(
                Fluid(id=f["id"], name=f["name"], kind=FluidKind(f["kind"])) for f in data["fluids"]
            ),
            targets=tuple(
                Target(
                    id=t["id"],
                    experiment_id=t["experiment_id"],
                    composition=Composition(
                        tuple(t["composition"]["num"]), t["composition"]["den"]
                    ),
                    tolerance=Rational(t["tolerance"]["num"], t["tolerance"]["den"]),
                    droplet_volume=t["droplet_volume"],
                    demand=t["demand"],
                )
                for t in data["targets"]
            ),
            chip=ChipLimits.from_dict(data["chip"]),
            step_limit=data["step_limit"],
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_json(text: str) -> Instance:
        return Instance.from_dict(json.loads(text))
