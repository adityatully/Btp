"""Exact rationals and composition vectors on the probability simplex (SPEC 5.1).

`Rational` is a tiny exact fraction (reduced, positive denominator) used for tolerances,
distances and costs. `Composition` is a point on the simplex over the F fluids of an instance:
integer numerators over one integer denominator, `sum(num) == den`, reduced by gcd. No floats
are used anywhere except in `to_float_array` / `__float__`, which exist only for observations
and rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Rational:
    """Exact fraction `num / den` with `den > 0` and `gcd(num, den) == 1`."""

    num: int
    den: int

    def __post_init__(self) -> None:
        if self.den <= 0:
            raise ValueError(f"Rational denominator must be positive, got {self.den}")
        if gcd(self.num, self.den) != 1:
            raise ValueError(f"Rational {self.num}/{self.den} is not reduced; use Rational.of")

    @staticmethod
    def of(num: int, den: int = 1) -> Rational:
        """Build a reduced rational from any integer numerator and nonzero denominator."""
        if den == 0:
            raise ZeroDivisionError("Rational denominator is zero")
        if den < 0:
            num, den = -num, -den
        g = gcd(num, den)
        return Rational(num // g, den // g)

    @staticmethod
    def parse(value: object) -> Rational:
        """Parse a config value exactly.

        Accepts an int, a string `"a/b"`, a decimal string `"0.5"`, or a float. A float is read
        through its shortest decimal repr (`0.1` -> 1/10), never through its binary value.
        """
        if isinstance(value, bool):
            raise TypeError(f"Cannot read a bool as a rational: {value!r}")
        if isinstance(value, int):
            return Rational.of(value)
        if isinstance(value, float):
            return Rational.parse(repr(value))
        if isinstance(value, str):
            text = value.strip()
            if "/" in text:
                a, b = text.split("/", 1)
                return Rational.of(int(a), int(b))
            if any(ch in text for ch in "eEnN"):
                raise ValueError(f"Rational value must be plain decimal or a/b, got {value!r}")
            sign = -1 if text.startswith("-") else 1
            text = text.lstrip("+-")
            whole, _, frac = text.partition(".")
            return Rational.of(sign * int((whole or "0") + frac), 10 ** len(frac))
        raise TypeError(f"Cannot read {type(value).__name__} as a rational: {value!r}")

    def __add__(self, other: Rational | int) -> Rational:
        o = _as_rational(other)
        return Rational.of(self.num * o.den + o.num * self.den, self.den * o.den)

    __radd__ = __add__

    def __sub__(self, other: Rational | int) -> Rational:
        o = _as_rational(other)
        return Rational.of(self.num * o.den - o.num * self.den, self.den * o.den)

    def __mul__(self, other: Rational | int) -> Rational:
        o = _as_rational(other)
        return Rational.of(self.num * o.num, self.den * o.den)

    __rmul__ = __mul__

    def __neg__(self) -> Rational:
        return Rational(-self.num, self.den)

    def __lt__(self, other: Rational | int) -> bool:
        o = _as_rational(other)
        return self.num * o.den < o.num * self.den

    def __le__(self, other: Rational | int) -> bool:
        o = _as_rational(other)
        return self.num * o.den <= o.num * self.den

    def __gt__(self, other: Rational | int) -> bool:
        o = _as_rational(other)
        return self.num * o.den > o.num * self.den

    def __ge__(self, other: Rational | int) -> bool:
        o = _as_rational(other)
        return self.num * o.den >= o.num * self.den

    def __float__(self) -> float:
        """Float view, for observations and rewards only."""
        return self.num / self.den

    def __str__(self) -> str:
        return str(self.num) if self.den == 1 else f"{self.num}/{self.den}"


ZERO = Rational(0, 1)


def _as_rational(x: Rational | int) -> Rational:
    if isinstance(x, Rational):
        return x
    if isinstance(x, int) and not isinstance(x, bool):
        return Rational(x, 1)
    raise TypeError(f"Expected Rational or int, got {type(x).__name__}")


@dataclass(frozen=True, slots=True, order=True)
class Composition:
    """Exact composition: fluid f has share `num[f] / den`.

    Invariants: all `num[f] >= 0`, `den > 0`, `sum(num) == den`, `gcd(*num, den) == 1`.
    Ordering is lexicographic on `(num, den)`; it is total and only used for canonical sorting.
    """

    num: tuple[int, ...]
    den: int

    def __post_init__(self) -> None:
        if not self.num:
            raise ValueError("Composition needs at least one fluid")
        if any(n < 0 for n in self.num):
            raise ValueError(f"Composition numerators must be >= 0, got {self.num}")
        if sum(self.num) != self.den:
            raise ValueError(f"Composition numerators {self.num} do not sum to den {self.den}")
        if gcd(*self.num, self.den) != 1:
            raise ValueError(f"Composition {self.num}/{self.den} is not reduced; use from_ints")

    @property
    def num_fluids(self) -> int:
        return len(self.num)

    @staticmethod
    def pure(fluid_id: int, num_fluids: int) -> Composition:
        """Composition of reservoir `fluid_id` (share 1 of that fluid)."""
        if not 0 <= fluid_id < num_fluids:
            raise ValueError(f"fluid_id {fluid_id} out of range for {num_fluids} fluids")
        return Composition(tuple(1 if f == fluid_id else 0 for f in range(num_fluids)), 1)

    @staticmethod
    def from_ints(num: tuple[int, ...] | list[int], den: int) -> Composition:
        """Reduce `num / den` to canonical form. `sum(num)` must equal `den`."""
        num = tuple(num)
        if den <= 0 or sum(num) != den:
            raise ValueError(f"Composition numerators {num} must sum to positive den {den}")
        g = gcd(*num, den)
        return Composition(tuple(n // g for n in num), den // g)

    @staticmethod
    def from_ratio(parts: tuple[int, ...] | list[int]) -> Composition:
        """From integer ratio parts, e.g. (2, 1, 1) -> 2/4, 1/4, 1/4."""
        return Composition.from_ints(tuple(parts), sum(parts))

    def share(self, fluid_id: int) -> Rational:
        return Rational.of(self.num[fluid_id], self.den)

    def to_float_array(self) -> npt.NDArray[np.float64]:
        """Float view, for network observations only."""
        return np.asarray(self.num, dtype=np.float64) / self.den

    def __str__(self) -> str:
        return "(" + ", ".join(str(self.share(f)) for f in range(len(self.num))) + ")"


def mix(c1: Composition, k: int, c2: Composition, l: int) -> Composition:
    """Exact `(k * c1 + l * c2) / (k + l)` for integer volumes `k, l >= 0`, `k + l > 0`."""
    if len(c1.num) != len(c2.num):
        raise ValueError("Cannot mix compositions over different fluid sets")
    if k < 0 or l < 0 or k + l == 0:
        raise ValueError(f"Mix volumes must be >= 0 with a positive total, got {k}, {l}")
    # Shares: k*a_f/d1 + l*b_f/d2 over (k + l)  ->  common denominator (k + l) * d1 * d2.
    a = k * c2.den
    b = l * c1.den
    num = tuple(a * x + b * y for x, y in zip(c1.num, c2.num, strict=True))
    return Composition.from_ints(num, (k + l) * c1.den * c2.den)


def linf_distance(c1: Composition, c2: Composition) -> Rational:
    """Exact max over fluids of |c1_f - c2_f|."""
    if len(c1.num) != len(c2.num):
        raise ValueError("Cannot compare compositions over different fluid sets")
    diff = max(abs(x * c2.den - y * c1.den) for x, y in zip(c1.num, c2.num, strict=True))
    return Rational.of(diff, c1.den * c2.den)
