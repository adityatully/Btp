"""Problem instances: `sample_instance(cfg, rng)` (SPEC 8.1, milestone M4) and `dilution_grid`.

`dilution_grid` builds every single-target dilution n/D of one sample with one buffer; it is the
"sweep all targets at an accuracy level" set of the reporting conventions, and the instance set
used for profiling until the sampler exists.
"""

from __future__ import annotations

from meda.core.chip import ChipLimits
from meda.core.composition import ZERO, Composition, Rational
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target


def dilution_grid(
    chip: ChipLimits,
    denominator: int,
    step_limit: int,
    demand: int = 1,
    droplet_volume: int = 1,
    tolerance: Rational = ZERO,
) -> list[Instance]:
    """All targets n/denominator (0 < n < denominator, reduced duplicates removed) of sample S
    diluted with buffer B, one target per instance."""
    if denominator < 2:
        raise ValueError(f"dilution_grid: denominator must be >= 2, got {denominator}")
    fluids = (Fluid(0, "S", FluidKind.SAMPLE), Fluid(1, "B", FluidKind.BUFFER))
    seen: set[Composition] = set()
    out = []
    for n in range(1, denominator):
        comp = Composition.from_ints((n, denominator - n), denominator)
        if comp in seen:
            continue
        seen.add(comp)
        target = Target(0, 0, comp, tolerance, droplet_volume, demand)
        out.append(
            Instance(
                f"grid-{chip.name}-d{denominator}-{comp.num[0]}of{comp.den}",
                fluids,
                (target,),
                chip,
                step_limit,
                {"generator": "dilution_grid", "denominator": denominator},
            )
        )
    return out
