"""Actions `Mix`, `Output`, `Discard`, sources, and uid-free canonical keys (SPEC 5.5).

A `Source` is a reservoir (by fluid id) or a droplet on chip (by uid). Canonical keys replace
droplet uids with the droplet's `(composition, volume)`, so they mean the same thing in any
state with the same canonical key. Tabular methods and caches key on them, never on a list index.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from meda.core.state import CanonicalKey, PrepState


class SourceKind(Enum):
    RESERVOIR = 0
    DROPLET = 1


@dataclass(frozen=True, slots=True)
class Source:
    """A reservoir (`index` = fluid id) or a droplet on chip (`index` = uid)."""

    kind: SourceKind
    index: int

    @staticmethod
    def reservoir(fluid_id: int) -> Source:
        return Source(SourceKind.RESERVOIR, fluid_id)

    @staticmethod
    def droplet(uid: int) -> Source:
        return Source(SourceKind.DROPLET, uid)

    @property
    def is_reservoir(self) -> bool:
        return self.kind is SourceKind.RESERVOIR

    def canonical_key(self, state: PrepState) -> CanonicalKey:
        """`("D", composition, volume)` for a droplet, `("R", fluid_id)` for a reservoir."""
        if self.is_reservoir:
            return ("R", self.index)
        d = state.droplet(self.index)
        return ("D", d.composition, d.volume)


@dataclass(frozen=True, slots=True)
class Mix:
    """Take `k` units from `a` and `l` units from `b`; make one droplet of volume k + l.

    Mix(a, k, b, l) and Mix(b, l, a, k) are the same action and have the same canonical key.
    """

    a: Source
    k: int
    b: Source
    l: int

    def canonical_key(self, state: PrepState) -> CanonicalKey:
        pa = (self.a.canonical_key(state), self.k)
        pb = (self.b.canonical_key(state), self.l)
        lo, hi = (pa, pb) if _key_le(pa, pb) else (pb, pa)
        return ("mix", lo, hi)


@dataclass(frozen=True, slots=True)
class Output:
    """Deliver `droplet_volume(target)` units of `src` to `target`.

    `src` is usually a droplet; a reservoir is allowed when the target is that pure fluid
    within tolerance.
    """

    src: Source
    target: int

    def canonical_key(self, state: PrepState) -> CanonicalKey:
        return ("output", self.src.canonical_key(state), self.target)


@dataclass(frozen=True, slots=True)
class Discard:
    """Send the whole droplet `uid` to waste."""

    uid: int

    def canonical_key(self, state: PrepState) -> CanonicalKey:
        return ("discard", Source.droplet(self.uid).canonical_key(state))


Action = Mix | Output | Discard


def _key_le(x: CanonicalKey, y: CanonicalKey) -> bool:
    # Keys are tuples of str/int/Composition; all are mutually ordered where they can meet.
    return bool(x <= y)
