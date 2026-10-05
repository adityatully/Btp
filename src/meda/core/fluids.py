"""`Fluid` and `FluidKind` (SPEC 5.1).

Each fluid has one reservoir on the chip. Reservoirs are implicit sources with unlimited volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FluidKind(Enum):
    """Fluid class; the cost weight of a reagent unit depends on it (SPEC 6.3)."""

    SAMPLE = "SAMPLE"
    REAGENT = "REAGENT"
    BUFFER = "BUFFER"


@dataclass(frozen=True, slots=True)
class Fluid:
    """One fluid of an instance. `id` is its index 0..F-1 within the instance."""

    id: int
    name: str
    kind: FluidKind
