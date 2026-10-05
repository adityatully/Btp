"""`Droplet` and `PrepState`: immutable, hashable state (SPEC 5.4).

Reservoirs are not droplets: each fluid f is an implicit source with unlimited volume and
composition `pure(f)`, and never counts toward storage.

Ids: droplet uids and DAG node ids share one counter, `PrepState.next_uid`. Reservoir f is DAG
node f, so the counter starts at F. A droplet made by a Mix has `uid == dag_node`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from meda.core.composition import Composition
from meda.core.problem import Instance

CanonicalKey = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class Droplet:
    """A droplet on chip. `volume` is in volume units; `dag_node` is the node that made it."""

    uid: int
    composition: Composition
    volume: int
    dag_node: int

    def sort_key(self) -> tuple[Composition, int, int]:
        return (self.composition, self.volume, self.uid)


@dataclass(frozen=True, slots=True)
class PrepState:
    """Decision state of sample preparation.

    `droplets` is sorted canonically (composition, volume, uid). `remaining_demand[t]` is the
    number of droplets target t still needs. `step` counts actions taken.
    """

    droplets: tuple[Droplet, ...]
    remaining_demand: tuple[int, ...]
    step: int
    next_uid: int

    def __post_init__(self) -> None:
        keys = [d.sort_key() for d in self.droplets]
        if keys != sorted(keys):
            raise ValueError("PrepState droplets must be sorted canonically; use PrepState.make")
        if any(d.volume < 1 for d in self.droplets):
            raise ValueError("PrepState droplets must have volume >= 1")
        if any(r < 0 for r in self.remaining_demand):
            raise ValueError("PrepState remaining_demand must be >= 0")

    @staticmethod
    def make(
        droplets: Iterable[Droplet], remaining_demand: Iterable[int], step: int, next_uid: int
    ) -> PrepState:
        """Build a state, sorting droplets canonically."""
        return PrepState(
            tuple(sorted(droplets, key=Droplet.sort_key)), tuple(remaining_demand), step, next_uid
        )

    @staticmethod
    def initial(instance: Instance) -> PrepState:
        """Empty chip, full demand, step 0. Ids start after the F reservoir nodes."""
        return PrepState((), tuple(t.demand for t in instance.targets), 0, instance.num_fluids)

    def droplet(self, uid: int) -> Droplet:
        for d in self.droplets:
            if d.uid == uid:
                return d
        raise KeyError(f"no droplet with uid {uid} on chip")

    def has_droplet(self, uid: int) -> bool:
        return any(d.uid == uid for d in self.droplets)

    @property
    def is_success(self) -> bool:
        return all(r == 0 for r in self.remaining_demand)

    def canonical_key(self, include_step: bool = False) -> CanonicalKey:
        """Multiset of (composition, volume) plus remaining demand; uids are dropped.

        Two states with equal keys are the same decision situation. Include `step` only when
        the step limit can bind.
        """
        droplets = tuple((d.composition, d.volume) for d in self.droplets)
        if include_step:
            return (droplets, self.remaining_demand, self.step)
        return (droplets, self.remaining_demand)
