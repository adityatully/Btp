"""`ChipLimits`: physical limits loaded from `configs/chip/*.yaml` (SPEC 5.2).

All volumes are integers in volume units. Every field is an open physical decision (SPEC 15)
and must come from config, never from a constant in code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class ChipLimits:
    """Physical limits of the chip for sample preparation.

    - `max_mix_volume` (M): a mix of k + l units is legal only if k + l <= M.
    - `min_mix_volume`: and only if k + l >= this.
    - `unit_volume`: size of one volume unit. Informational: all volumes in the core are
      integer counts of this unit.
    - `min_droplet_volume`: smallest droplet that may remain after a partial take.
    - `allow_partial_take`: a mix or output may take fewer units than a droplet holds.
    - `storage_capacity`: max number of non-reservoir droplets on chip after any action.
    - `max_denominator`: mixes whose result denominator exceeds this are illegal.
    - `allow_same_composition_merge`: allow mixing two droplets of equal composition.
    """

    name: str
    max_mix_volume: int
    min_mix_volume: int
    unit_volume: int
    min_droplet_volume: int
    allow_partial_take: bool
    storage_capacity: int
    max_denominator: int
    allow_same_composition_merge: bool

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            expected: type = {
                "name": str,
                "allow_partial_take": bool,
                "allow_same_composition_merge": bool,
            }.get(f.name, int)
            if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
                raise TypeError(
                    f"chip '{self.name}': {f.name} must be {expected.__name__}, got {value!r}"
                )
        if self.min_mix_volume < 2:
            raise ValueError(f"chip '{self.name}': min_mix_volume must be >= 2 (k, l >= 1)")
        if self.max_mix_volume < self.min_mix_volume:
            raise ValueError(f"chip '{self.name}': max_mix_volume < min_mix_volume")
        for name in ("unit_volume", "min_droplet_volume", "storage_capacity", "max_denominator"):
            if getattr(self, name) < 1:
                raise ValueError(f"chip '{self.name}': {name} must be >= 1")

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ChipLimits:
        """Build from a config mapping. Missing or unknown keys raise."""
        names = {f.name for f in fields(ChipLimits)}
        keys = set(data)
        if keys != names:
            raise ValueError(
                f"chip config keys mismatch: missing {sorted(names - keys)}, "
                f"unknown {sorted(keys - names)}"
            )
        return ChipLimits(**{k: data[k] for k in names})

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
