"""Fukui's reuse of intermediate droplets on DFMM (Fukui & Yamashita, "An Efficient Cost
Reduction Method By Reusing Intermediate Droplets in MEDA Biochips", ISOCC 2024; SPEC 10.4).

DFMM (`dfmm`) makes each target on its own. Fukui's method makes a new target P (ratio-sum PS)
from a droplet D_w left by an earlier mixing graph (ratio-sum S_w): e_w units of D_w are mixed
with e_n = es - e_w units of a new droplet D_n, D_n made by DFMM. D_n's ratio a (sum S_n) solves,
fluid by fluid, (d_j / S_w)(e_w / es) + (a_j / S_n)(e_n / es) = p_j / PS (the paper's Eq. 3).
Figs. 2-3: P 4:3:5 from D_w <3:0:1> at e_w : e_n = 1 : 2 needs D_n 1:3:4.

Our reading of what the two-page paper leaves open (`notes/lit_bench/selection.md`):
- How e_w / es is chosen is not stated. Ours tries every e_w < es <= M (e_w no more than D_w
  holds, D_n's ratio non-negative) and keeps the fewest reagent units (the paper's cost), then
  the least waste; a target made without reuse (plain DFMM) is kept when that is no dearer.
- Many targets: targets in order, each droplet of demand a target in sequence; a target may reuse
  one droplet left by any earlier graph (the paper reuses one, more being its future work). A
  leftover of the target's own ratio holding a droplet is taken as it is.
- D_n of a single fluid is that fluid, drawn directly. Exact targets: DFMM takes the ratio as
  given, so a tolerant target is made at its stated ratio.
"""

from __future__ import annotations

from functools import cache
from math import gcd

from meda.core.composition import Composition
from meda.core.problem import Instance
from meda.solvers.baselines.dfmm import best_graph
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph

Key = tuple[int, int, int]  # reagent units, waste units, mixes


def partner(p: Composition, d: Composition, e_w: int, es: int) -> Composition | None:
    """D_n's ratio when e_w of es units are D_w (Eq. 3), or None when a share is negative."""
    e_n = es - e_w
    den = p.den * d.den // gcd(p.den, d.den)
    nums = [
        pj * (den // p.den) * es - dj * (den // d.den) * e_w
        for pj, dj in zip(p.num, d.num, strict=True)
    ]
    if min(nums) < 0:
        return None
    return Composition.from_ints(nums, den * e_n)


class Fukui:
    def __init__(self, num_fluids: int, lo: int, hi: int) -> None:
        self.num_fluids, self.lo, self.hi = num_fluids, lo, hi
        self.nodes: list[tuple[In, ...]] = []
        self.comps: list[Composition] = []
        self.left: list[int] = []

    @cache  # noqa: B019 - one Fukui per instance
    def dfmm(self, ratio: tuple[int, ...]) -> tuple[PaperGraph, Key] | None:
        try:
            g = best_graph(ratio, self.lo, self.hi)
        except NotApplicable:
            return None
        root = g.volume(len(g.nodes) - 1)
        c = g.counts(root)
        drawn, nw = c["drawn"], c["nw"]
        assert isinstance(drawn, tuple) and isinstance(nw, int)
        return g, (sum(drawn), nw, len(g.nodes))

    def droplet(self, p: Composition, v: int) -> int:
        """A node holding v units of p for delivery."""
        for i, (c, u) in enumerate(zip(self.comps, self.left, strict=True)):
            if c == p and u >= v:
                self.left[i] -= v
                return i
        best: tuple[Key, int | None, int, int, Composition | None] | None = None
        plain = self.dfmm(p.num)
        if plain is not None and plain[0].volume(len(plain[0].nodes) - 1) >= v:
            root = plain[0].volume(len(plain[0].nodes) - 1)
            k = plain[1]
            best = ((k[0], k[1] + root - v, k[2]), None, 0, 0, None)
        for w, (d, u) in enumerate(zip(self.comps, self.left, strict=True)):
            if u == 0 or d == p:
                continue
            for es in range(max(2, self.lo, v), self.hi + 1):
                for e_w in range(1, min(u, es - 1) + 1):
                    dn = partner(p, d, e_w, es)
                    if dn is None or dn == d:
                        continue
                    key = self._price(dn, es - e_w, es - v)
                    if key is not None and (best is None or key < best[0]):
                        best = (key, w, e_w, es, dn)
        if best is None:
            raise NotApplicable(f"{p}: neither DFMM nor one reuse makes it within M {self.hi}")
        _, reused, e_w, es, dn = best
        if reused is None:
            assert plain is not None
            root = self._add(plain[0])
            self.left[root] -= v
            return root
        assert dn is not None
        if dn.den == 1:
            ins = [In(es - e_w, fluid=dn.num.index(1))]
        else:
            got = self.dfmm(dn.num)
            assert got is not None
            sub = self._add(got[0])
            self.left[sub] -= es - e_w
            ins = [In(es - e_w, node=sub)]
        self.left[reused] -= e_w
        return self._node((In(e_w, node=reused), *ins), es - v)

    def _price(self, dn: Composition, e_n: int, spill: int) -> Key | None:
        if dn.den == 1:
            return (e_n, spill, 1)
        got = self.dfmm(dn.num)
        if got is None:
            return None
        g, (units, nw, nm) = got
        root = g.volume(len(g.nodes) - 1)
        if root < e_n:
            return None
        return (units, nw + root - e_n + spill, nm + 1)

    def _add(self, g: PaperGraph) -> int:
        base = len(self.nodes)
        for node in g.nodes:
            ins = tuple(In(x.units, node=x.node + base) if x.node is not None else x for x in node)
            for x in ins:
                if x.node is not None:
                    self.left[x.node] -= x.units
            self._node(ins, None)
        return len(self.nodes) - 1

    def _node(self, ins: tuple[In, ...], left: int | None) -> int:
        self.nodes.append(ins)
        self.comps.append(PaperGraph(self.num_fluids, tuple(self.nodes)).compositions()[-1])
        vol = sum(x.units for x in ins)
        self.left.append(vol if left is None else left)
        return len(self.nodes) - 1


def fukui(instance: Instance) -> PaperGraph:
    chip = instance.chip
    f = Fukui(instance.num_fluids, chip.min_mix_volume, chip.max_mix_volume)
    roots: list[int] = []
    for t in instance.targets:
        if t.composition.den == 1:
            raise NotApplicable(f"target {t.id} is a pure fluid")
        for _ in range(t.demand):
            roots.append(f.droplet(t.composition, t.droplet_volume))
    return PaperGraph(instance.num_fluids, tuple(f.nodes), tuple(roots))
