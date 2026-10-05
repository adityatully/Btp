"""`StepCost`, `CostWeights` and exact cost accounting (SPEC section 6).

Weights are exact rationals. Config values may be ints, decimals (`0.5`) or strings (`"1/2"`);
they are parsed exactly by `Rational.parse`, so the scalar cost is an exact `Rational` and exact
search compares plan costs without rounding. Only the env converts cost to a float reward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from meda.core.composition import ZERO, Rational
from meda.core.fluids import Fluid, FluidKind


@dataclass(frozen=True, slots=True)
class StepCost:
    """Cost components of one step (or, summed, of an episode). All exact integers.

    - `reagent_by_fluid[f]`: units taken from reservoir f.
    - `mix_ops`: number of Mix actions.
    - `waste_volume`: units sent to waste (Discard and leftovers at the end).
    - `outputs`: droplets delivered.
    """

    reagent_by_fluid: tuple[int, ...]
    mix_ops: int
    waste_volume: int
    outputs: int

    @staticmethod
    def zero(num_fluids: int) -> StepCost:
        return StepCost((0,) * num_fluids, 0, 0, 0)

    def __add__(self, other: StepCost) -> StepCost:
        return StepCost(
            tuple(
                a + b for a, b in zip(self.reagent_by_fluid, other.reagent_by_fluid, strict=True)
            ),
            self.mix_ops + other.mix_ops,
            self.waste_volume + other.waste_volume,
            self.outputs + other.outputs,
        )

    @property
    def total_reagent(self) -> int:
        return sum(self.reagent_by_fluid)

    def volume_of_kind(self, fluids: Sequence[Fluid], kind: FluidKind) -> int:
        """Units taken from reservoirs of `kind` (e.g. sample volume for SAMPLE)."""
        return sum(v for f, v in zip(fluids, self.reagent_by_fluid, strict=True) if f.kind is kind)


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Weights of the scalar cost (SPEC 6.3):

    cost = sum_f w_fluid[kind(f)] * reagent_by_fluid[f] + w_mix * mix_ops + w_waste * waste
    """

    w_sample: Rational
    w_reagent: Rational
    w_buffer: Rational
    w_mix: Rational
    w_waste: Rational

    def __post_init__(self) -> None:
        for name in ("w_sample", "w_reagent", "w_buffer", "w_mix", "w_waste"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"cost weight {name} must be >= 0 (search needs nonnegative costs)"
                )

    def fluid_weight(self, kind: FluidKind) -> Rational:
        if kind is FluidKind.SAMPLE:
            return self.w_sample
        if kind is FluidKind.REAGENT:
            return self.w_reagent
        return self.w_buffer

    def scalar(self, cost: StepCost, fluids: Sequence[Fluid]) -> Rational:
        """Exact scalar cost of `cost` for an instance with these `fluids`."""
        total = ZERO
        for f, units in zip(fluids, cost.reagent_by_fluid, strict=True):
            if units:
                total = total + self.fluid_weight(f.kind) * units
        return total + self.w_mix * cost.mix_ops + self.w_waste * cost.waste_volume

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> CostWeights:
        """From the `costs` config: `{w_fluid: {SAMPLE, REAGENT, BUFFER}, w_mix, w_waste}`."""
        if set(data) != {"w_fluid", "w_mix", "w_waste"}:
            raise ValueError(f"costs config must have w_fluid, w_mix, w_waste; got {sorted(data)}")
        wf = data["w_fluid"]
        kinds = {k.value for k in FluidKind}
        if set(wf) != kinds:
            raise ValueError(f"costs.w_fluid must have keys {sorted(kinds)}; got {sorted(wf)}")
        return CostWeights(
            w_sample=Rational.parse(wf["SAMPLE"]),
            w_reagent=Rational.parse(wf["REAGENT"]),
            w_buffer=Rational.parse(wf["BUFFER"]),
            w_mix=Rational.parse(data["w_mix"]),
            w_waste=Rational.parse(data["w_waste"]),
        )
