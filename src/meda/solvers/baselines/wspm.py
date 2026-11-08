"""WSPM: droplet-size-aware dilution on MEDA, with primary-droplet sharing and incremental
preparation (Li, Lai, Yu, Chakrabarty, Ho & Lee, "Droplet Size-Aware and Error-Correcting
Sample Preparation Using Micro-Electrode-Dot-Array Digital Microfluidic Biochips", IEEE TBioCAS
11(6), 2017, Section IV; SPEC 10.4).

A droplet is a concentration C (numerator over 1/delta) and a size S <= Sc. A MEDA dilution
mixes D_i(C_i, S_i) and D_j(C_j, S_j) and keeps a droplet of any integer size up to S_i + S_j
(pseudo-dispensing); the rest is waste. Primary droplets (PDs) are the sample diluted only with
buffer (Fig. 8, found by search). The dilution graph is built top-down: a droplet is a PD, or
the mix of a PD (left) and a droplet of lower concentration (right), chosen by least sample,
then operations, then waste (Fig. 9). PD sharing takes a PD from the waste of another PD's
chain (Section IV-D); incremental preparation takes any droplet from what earlier targets left
(Section IV-E). Example (Fig. 10): 80/256 from PDs 128(1) and 32(1), 128(1) being 32(1)'s waste.

Our reading of what the paper leaves open (`notes/lit_bench/selection.md`):
- The paper rounds every mix to the nearest 1/delta (its Eq. 1), which our exact engine cannot
  do: only dilutions whose result is exact at the targets' accuracy are taken, so some of
  WSPM's options are lost (marked in the paper log). delta is 1 over the targets' common
  denominator; a tolerant target is first moved to the CF of smallest denominator within it.
- The estimated cost table (Eqs. 3-5) and the greedy top-down choice are one exact recursion
  over (C, S): a node's cost is its children's plus one operation and its leftover; it is at
  least as good as the paper's greedy choice with estimated costs.
- PD sharing and incremental reuse are greedy: before a droplet is built, the largest leftover
  of its concentration that holds its size is taken (the paper's brute force over PD subsets is
  replaced by this); demand d is d targets in sequence (incremental).
- Sample cost counts sample units; Sc is the chip's M; buffer is free, as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import gcd

from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, sample_and_buffer

Cost = tuple[int, int, int]  # sample units, operations, waste units
INF: Cost = (10**9, 10**9, 10**9)


def add(*costs: Cost) -> Cost:
    return (sum(c[0] for c in costs), sum(c[1] for c in costs), sum(c[2] for c in costs))


@dataclass(frozen=True, slots=True)
class Recipe:
    """How droplet (c, s) is made: from the sample reservoir, by a primary dilution of
    (c_in, s_in) with s_b buffer, or by mixing PD (c1, s1) with droplet (c2, s2)."""

    kind: str  # "sample", "primary", "mix"
    a: tuple[int, int] = (0, 0)
    b: tuple[int, int] = (0, 0)


class Tables:
    def __init__(self, den: int, sc: int) -> None:
        self.den, self.sc = den, sc
        self.pd: dict[tuple[int, int], tuple[Cost, Recipe]] = {}
        frontier = []
        for s in range(1, sc + 1):
            self.pd[(den, s)] = ((s, 0, 0), Recipe("sample"))
            frontier.append((den, s))
        while frontier:  # concentrations only fall: a pass per round of primary dilutions
            nxt = []
            for c_in, s_in in sorted(frontier, reverse=True):
                base = self.pd[(c_in, s_in)][0]
                for s_b in range(1, sc - s_in + 1):
                    num = c_in * s_in
                    if num % (s_in + s_b):
                        continue
                    c = num // (s_in + s_b)
                    if c == 0:
                        continue
                    for s in range(1, s_in + s_b + 1):
                        cost = add(base, (0, 1, s_in + s_b - s))
                        if cost < self.pd.get((c, s), (INF, Recipe("sample")))[0]:
                            self.pd[(c, s)] = (cost, Recipe("primary", (c_in, s_in), (0, s_b)))
                            nxt.append((c, s))
            frontier = nxt
        self.pds = sorted(self.pd)

    @cache  # noqa: B019 - one Tables per instance
    def best(self, c: int, s: int) -> tuple[Cost, Recipe]:
        if c == 0:
            return (0, 0, 0), Recipe("buffer")
        out = self.pd.get((c, s), (INF, Recipe("sample")))
        for c1, s1 in self.pds:
            if c1 == c:
                continue
            for s2 in range(1, self.sc - s1 + 1):
                if s > s1 + s2:
                    continue
                num = c * (s1 + s2) - s1 * c1
                if num < 0 or num % s2:
                    continue
                c2 = num // s2
                if c2 >= c or c2 == c1:
                    continue
                right = self.best(c2, s2)[0]
                if right == INF:
                    continue
                cost = add(self.pd[(c1, s1)][0], right, (0, 1, s1 + s2 - s))
                if cost < out[0]:
                    out = (cost, Recipe("mix", (c1, s1), (c2, s2)))
        return out


class Builder:
    """Emits droplets as PaperGraph nodes, taking leftovers (PD sharing, incremental) first."""

    def __init__(self, t: Tables, sample: int, buffer: int, fluids: int) -> None:
        self.t, self.sample, self.buffer, self.fluids = t, sample, buffer, fluids
        self.nodes: list[tuple[In, ...]] = []
        self.left: dict[int, list[list[int]]] = {}  # c -> [[node, units left]]

    def droplet(self, c: int, s: int) -> In:
        if c == 0:
            return In(s, fluid=self.buffer)
        if c == self.t.den:
            return In(s, fluid=self.sample)
        spare = [cell for cell in self.left.get(c, []) if cell[1] >= s]
        if spare:
            cell = max(spare, key=lambda x: (x[1], -x[0]))
            cell[1] -= s
            return In(s, node=cell[0])
        cost, how = self.t.best(c, s)
        reuse = self._from_leftovers(c, s)
        if reuse is not None and reuse[0] < cost:
            _, (cell, s1), (c2, s2) = reuse
            cell[1] -= s1
            ins = (In(s1, node=cell[0]), self.droplet(c2, s2))
            self.nodes.append(ins)
            node = len(self.nodes) - 1
            self.left.setdefault(c, []).append([node, s1 + s2 - s])
            return In(s, node=node)
        if how.kind == "primary":
            (c_in, s_in), (_, s_b) = how.a, how.b
            ins = (self.droplet(c_in, s_in), In(s_b, fluid=self.buffer))
            vol = s_in + s_b
        elif how.kind == "mix":
            ins = (self.droplet(*how.a), self.droplet(*how.b))
            vol = how.a[1] + how.b[1]
        else:
            raise NotApplicable(f"no exact dilution makes {c}/{self.t.den} in {s} units")
        self.nodes.append(ins)
        node = len(self.nodes) - 1
        self.left.setdefault(c, []).append([node, vol - s])
        return In(s, node=node)

    def _from_leftovers(
        self, c: int, s: int
    ) -> tuple[Cost, tuple[list[int], int], tuple[int, int]] | None:
        """The cheapest mix of a leftover droplet (a pseudo primary droplet, free) with any
        droplet of lower concentration, as incremental preparation allows."""
        best: tuple[Cost, tuple[list[int], int], tuple[int, int]] | None = None
        for c1, cells in sorted(self.left.items()):
            for cell in cells:
                for s1 in range(1, min(cell[1], self.t.sc - 1) + 1):
                    for s2 in range(1, self.t.sc - s1 + 1):
                        if s > s1 + s2:
                            continue
                        num = c * (s1 + s2) - s1 * c1
                        if num < 0 or num % s2:
                            continue
                        c2 = num // s2
                        if c2 == c1 or (c2 >= c and c1 > c):
                            continue
                        right = self._price(c2, s2, cell)
                        if right == INF:
                            continue
                        cost = add(right, (0, 1, s1 + s2 - s))
                        if best is None or cost < best[0]:
                            best = (cost, (cell, s1), (c2, s2))
        return best

    def _price(self, c: int, s: int, taken: list[int]) -> Cost:
        if c == 0:
            return (0, 0, 0)
        if c == self.t.den:
            return (s, 0, 0)
        if any(cell is not taken and cell[1] >= s for cell in self.left.get(c, [])):
            return (0, 0, 0)
        return self.t.best(c, s)[0]


def wspm(instance: Instance) -> PaperGraph:
    s, b = sample_and_buffer(instance)
    sc = instance.chip.max_mix_volume
    cfs = [_exact(instance, t.id, s) for t in instance.targets]
    den = 1
    for c in cfs:
        den = den * c.den // gcd(den, c.den)
    if den > instance.chip.max_denominator:
        raise NotApplicable(f"the targets' common denominator {den} is above the chip's limit")
    tables = Tables(den, sc)
    build = Builder(tables, s, b, instance.num_fluids)
    roots: list[int] = []
    for c, t in zip(cfs, instance.targets, strict=True):
        num = c.num * (den // c.den)
        if num in (0, den):
            raise NotApplicable(f"target {t.id} is a pure fluid")
        for _ in range(t.demand):
            got = build.droplet(num, t.droplet_volume)
            if got.node is None:
                raise AssertionError("a target is a mix")
            roots.append(got.node)
    return PaperGraph(instance.num_fluids, tuple(build.nodes), tuple(roots))


def _exact(instance: Instance, target: int, s: int) -> Rational:
    t = instance.targets[target]
    c = t.composition.share(s)
    if t.tolerance.num == 0:
        return c
    for d in range(2, instance.chip.max_denominator + 1):
        near = Rational.of((2 * c.num * d + c.den) // (2 * c.den), d)
        diff = near - c
        if max(diff, -diff) <= t.tolerance:
            return near
    raise NotApplicable(f"target {target}: no CF within tolerance under the chip's limit")
