"""Composition and Rational: canonical form, exact mixing, distances (SPEC 17.1)."""

from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from meda.core.composition import Composition, Rational, linf_distance, mix


@st.composite
def compositions(draw: st.DrawFn, num_fluids: int = 3) -> Composition:
    parts = draw(st.lists(st.integers(0, 50), min_size=num_fluids, max_size=num_fluids))
    if sum(parts) == 0:
        parts[0] = 1
    return Composition.from_ratio(parts)


def as_fractions(c: Composition) -> list[Fraction]:
    return [Fraction(n, c.den) for n in c.num]


volumes = st.integers(1, 20)


@given(compositions())
def test_canonical_form(c: Composition) -> None:
    assert sum(c.num) == c.den
    assert Fraction(c.num[0], c.den).denominator <= c.den
    assert Composition.from_ints(tuple(n * 7 for n in c.num), c.den * 7) == c


@given(compositions(), volumes, compositions(), volumes)
def test_mix_is_exact(c1: Composition, k: int, c2: Composition, l: int) -> None:
    expected = [
        (k * a + l * b) / (k + l) for a, b in zip(as_fractions(c1), as_fractions(c2), strict=True)
    ]
    assert as_fractions(mix(c1, k, c2, l)) == expected


@given(compositions(), volumes, compositions(), volumes)
def test_mix_symmetric(c1: Composition, k: int, c2: Composition, l: int) -> None:
    assert mix(c1, k, c2, l) == mix(c2, l, c1, k)


@given(compositions(), volumes, volumes)
def test_mix_with_itself_is_identity(c: Composition, k: int, l: int) -> None:
    assert mix(c, k, c, l) == c


@given(compositions(), compositions())
def test_linf_distance_exact(c1: Composition, c2: Composition) -> None:
    d = linf_distance(c1, c2)
    expected = max(abs(a - b) for a, b in zip(as_fractions(c1), as_fractions(c2), strict=True))
    assert Fraction(d.num, d.den) == expected


def test_constructors() -> None:
    assert Composition.pure(1, 3) == Composition((0, 1, 0), 1)
    assert Composition.from_ratio((2, 1, 1)) == Composition((2, 1, 1), 4)
    assert Composition.from_ints((2, 2), 4) == Composition((1, 1), 2)
    assert list(Composition.from_ratio((1, 3)).to_float_array()) == [0.25, 0.75]


@pytest.mark.parametrize(("num", "den"), [((1, 1), 3), ((2, 2), 4), ((-1, 2), 1), ((0, 0), 0)])
def test_invalid_compositions_raise(num: tuple[int, ...], den: int) -> None:
    with pytest.raises(ValueError):
        Composition(num, den)


def test_worked_example_from_project_context() -> None:
    # E1 = I:J at 3:2 with I = A+B 1:1, J = A+C 1:1; E3 = E1 + C at 1:1.
    a, b, c = (Composition.pure(f, 3) for f in range(3))
    i, j = mix(a, 1, b, 1), mix(a, 1, c, 1)
    e1 = mix(i, 3, j, 2)
    assert e1 == Composition.from_ratio((5, 3, 2))
    assert mix(e1, 1, c, 1) == Composition.from_ratio((25, 15, 60))


# --- Rational


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, (1, 2)),
        (0.1, (1, 10)),
        ("1/2", (1, 2)),
        ("3/6", (1, 2)),
        (10, (10, 1)),
        (10.0, (10, 1)),
        ("2.25", (9, 4)),
        ("-0.5", (-1, 2)),
    ],
)
def test_rational_parse(value: object, expected: tuple[int, int]) -> None:
    assert Rational.parse(value) == Rational(*expected)


@pytest.mark.parametrize("value", [True, "1e-3", "nan", None, [1]])
def test_rational_parse_rejects(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Rational.parse(value)


fracs = st.fractions(max_denominator=1000)


@given(fracs, fracs)
def test_rational_arithmetic_matches_fraction(x: Fraction, y: Fraction) -> None:
    a = Rational.of(x.numerator, x.denominator)
    b = Rational.of(y.numerator, y.denominator)
    for got, want in [(a + b, x + y), (a - b, x - y), (a * b, x * y)]:
        assert Fraction(got.num, got.den) == want
    assert (a < b) == (x < y)
    assert (a <= b) == (x <= y)
    assert (a == b) == (x == y)
