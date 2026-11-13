"""Execution platform: resource counts and durations for the harness (SPEC 14.2).

Every number is a physical assumption (rule 8), read from `configs/platform/*.yaml`. Time is an
integer number of ticks; `tick_ms` only converts ticks for reporting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PLATFORM_DIR = Path(__file__).resolve().parents[3] / "configs" / "platform"

# field -> smallest allowed value
_MIN = {
    "tick_ms": 1,
    "mixers": 1,
    "ports_per_fluid": 1,
    "detectors": 1,
    "dispense_ticks": 0,
    "mix_ticks": 1,
    "detect_ticks": 1,
    "changeover_ticks": 0,
}


@dataclass(frozen=True, slots=True)
class Platform:
    """Resource counts and operation durations (ticks). `storage_capacity` None means the
    instance's chip limit (`ChipLimits.storage_capacity`)."""

    name: str
    tick_ms: int
    mixers: int
    ports_per_fluid: int
    detectors: int
    storage_capacity: int | None
    dispense_ticks: int
    mix_ticks: int
    detect_ticks: int
    changeover_ticks: int

    def __post_init__(self) -> None:
        for name, low in _MIN.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < low:
                raise ValueError(f"platform '{self.name}': {name} must be an int >= {low}")
        cap = self.storage_capacity
        if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or cap < 1):
            raise ValueError(f"platform '{self.name}': storage_capacity must be null or >= 1")

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Platform:
        """From a config mapping. Missing or unknown keys raise."""
        names = {f.name for f in fields(Platform)}
        if set(data) != names:
            raise ValueError(
                f"platform config keys mismatch: missing {sorted(names - set(data))}, "
                f"unknown {sorted(set(data) - names)}"
            )
        return Platform(**{k: data[k] for k in names})

    @staticmethod
    def load(name: str = "meda_default", directory: Path = PLATFORM_DIR) -> Platform:
        """Read `configs/platform/<name>.yaml`."""
        data = OmegaConf.to_container(OmegaConf.load(directory / f"{name}.yaml"))
        if not isinstance(data, dict):
            raise ValueError(f"platform config {name}.yaml is not a mapping")
        return Platform.from_dict({str(k): v for k, v in data.items()})

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
