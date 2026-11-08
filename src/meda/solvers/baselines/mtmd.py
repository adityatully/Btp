"""MTCS, MTR and MTMD: 1:1 mixing forests with common subtrees and demand (Shalu, Kumar,
Singla, Roy, Chakrabarty, Chakrabarti & Bhattacharya, "Demand-Driven Single- and Multitarget
Mixture Preparation Using Digital Microfluidic Biochips", ACM TODAES 23(4), 2018; SPEC 10.4).

A target ratio a_1 : ... : a_N with sum 2^d is a Min-Mix tree: reagent x_j enters as a leaf at
every depth where a_j / 2^d has a 1 bit, and droplets of one depth are mixed 1:1 in pairs, each
pair making two droplets one depth up. MTCS (Algorithms 1-2) scans the bit columns from the
least significant bit up and pairs leaves that also appear together in a later column (its
AND-ed matrix), so one mix serves both: the second droplet of the mix replaces that pair of
leaves (a common subtree). Every mix's unused droplet is kept in a list U and taken wherever its
composition is needed. MTR (Algorithm 3) does the same over several targets, their columns
interleaved (target 1's first column, target 2's first column, ...). MTMD (Section 5) builds the
MTR forest for the distinct targets and then each further droplet of demand by cloning the
target's tree top-down, taking a node from U where one is kept and rebuilding it otherwise.
Example (Fig. 3): 7:7:5:5:3:3:2 gives Tms 9 mix-split steps and W 5 waste droplets.

Our reading of what the paper leaves open (`notes/lit_bench/selection.md`):
- The paper pairs at random (Algorithm 2). Ours is fixed: first a pair of leaves whose mix is in
  U is taken from U; then pairs of leaves that also meet in a later column (fewest later
  meetings first, then fluid order); then the rest in composition order, never two droplets of
  one composition while another choice exists (two such droplets make no new composition: one
  goes up, the other to U).
- W counts the droplets in U at the end except each tree root's second droplet, as the paper
  does (Fig. 3's Tms 9, W 5, Ncs 3 add up only that way); our waste counts every leftover.
- On MEDA every droplet is the targets' droplet volume v and a mix is v:v (2v <= M); targets of
  different volumes are not applicable. A ratio needs a power-of-two sum (the paper's targets
  are given at an accuracy level): a non-dyadic exact target is not applicable, a tolerant one
  is moved to the nearest k / 2^n within tolerance, n = log2 of the chip's denominator limit,
  ties to the smaller n then the smaller k.
- The paper assumes unlimited storage; a plan that overfills the chip is not applicable (`emit`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from meda.core.composition import Composition, mix
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph

Item = tuple[str, int]  # ("fluid", j) or ("node", i)


@dataclass
class Forest:
    """Mix nodes (two unit inputs each), their compositions, and the list U of spare droplets."""

    num_fluids: int
    nodes: list[tuple[Item, Item]] = field(default_factory=list)
    comps: list[Composition] = field(default_factory=list)
    spare: dict[Composition, list[int]] = field(default_factory=lambda: defaultdict(list))

    def comp(self, item: Item) -> Composition:
        kind, i = item
        return Composition.pure(i, self.num_fluids) if kind == "fluid" else self.comps[i]

    def take(self, comp: Composition) -> Item | None:
        if self.spare.get(comp):
            return ("node", self.spare[comp].pop(0))
        return None

    def mix(self, a: Item, b: Item) -> Item:
        ca, cb = self.comp(a), self.comp(b)
        if ca == cb:  # no new composition: one droplet goes up, the other is kept
            if b[0] == "node":
                self.spare[cb].append(b[1])
            return a
        self.nodes.append((a, b))
        self.comps.append(mix(ca, 1, cb, 1))
        self.spare[self.comps[-1]].append(len(self.nodes) - 1)
        return ("node", len(self.nodes) - 1)


def bits(ratio: tuple[int, ...]) -> tuple[int, list[list[int]]]:
    """(d, columns): column k (k = 0 is the least significant bit) lists the fluids with a 1."""
    total = sum(ratio)
    d = total.bit_length() - 1
    if total != 1 << d:
        raise NotApplicable(f"ratio sum {total} is not a power of two")
    return d, [[j for j, a in enumerate(ratio) if a >> k & 1] for k in range(d)]


def uniform(c: Composition) -> frozenset[int] | None:
    """The fluids of c if it is an equal mix of 2^m >= 2 of them, else None."""
    parts = [j for j, x in enumerate(c.num) if x]
    if len(parts) < 2 or len(set(c.num[j] for j in parts)) != 1 or len(parts) & (len(parts) - 1):
        return None
    return frozenset(parts)


def mtr(ratios: list[tuple[int, ...]], num_fluids: int) -> tuple[Forest, list[Item]]:
    """The MTR forest (MTCS for one ratio): the forest and each target's root droplet."""
    f = Forest(num_fluids)
    cols = [bits(r) for r in ratios]
    depth = max(d for d, _ in cols)
    order = [(k, i) for k in range(depth) for i in range(len(ratios)) if k < cols[i][0]]
    sets = [frozenset(cols[i][1][k]) for k, i in order]
    carried: list[dict[int, list[Item]]] = [defaultdict(list) for _ in ratios]
    for o, (k, i) in enumerate(order):
        leaves = set(cols[i][1][k])
        # 1. leaves replaced by a kept droplet that mixes exactly them (a common subtree):
        #    2^m leaves of column k are one droplet of column k + m
        for comp in sorted(f.spare, key=lambda c: (-len(uniform(c) or ()), c.num)):
            fluids = uniform(comp)
            while fluids is not None and fluids <= leaves and f.spare.get(comp):
                got = f.take(comp)
                assert got is not None
                leaves -= fluids
                carried[i][k + len(fluids).bit_length() - 1].append(got)
        items: list[Item] = [("fluid", j) for j in sorted(leaves)] + carried[i][k]
        carried[i][k + 1] += _pair(f, items, sets[o + 1 :])
    roots: list[Item] = []
    for i, (d, _) in enumerate(cols):
        if len(carried[i][d]) != 1:
            raise AssertionError("a Min-Mix tree ends in one root")
        roots.append(carried[i][d][0])
    return f, roots


def _pair(f: Forest, items: list[Item], later: list[frozenset[int]]) -> list[Item]:
    """Pair one column's droplets, first those whose mix a later column's leaves will want
    (largest first: MTCS's AND-ed matrix); the droplets one depth up."""
    if len(items) % 2:
        raise AssertionError("a Min-Mix column has an odd number of droplets")
    up: list[Item] = []
    rest = sorted(items, key=lambda it: (f.comp(it).den, f.comp(it).num))

    def wanted(a: Item, b: Item) -> int:
        ca, cb = f.comp(a), f.comp(b)
        if ca == cb:
            return 0
        fluids = uniform(mix(ca, 1, cb, 1))
        if fluids is None or not any(fluids <= s for s in later):
            return 0
        return len(fluids)

    while True:  # a pair whose mix is kept in U: take it, the pair's droplets go to U
        hit = next(
            (
                (x, y, got)
                for x, a in enumerate(rest)
                for y, b in enumerate(rest)
                if x < y
                and f.comp(a) != f.comp(b)
                and (got := f.spare.get(mix(f.comp(a), 1, f.comp(b), 1)))
            ),
            None,
        )
        if hit is None:
            break
        x, y, _ = hit
        for it in (rest[x], rest[y]):
            if it[0] == "node":
                f.spare[f.comps[it[1]]].append(it[1])
        took = f.take(mix(f.comp(rest[x]), 1, f.comp(rest[y]), 1))
        assert took is not None
        up.append(took)
        rest = [it for n, it in enumerate(rest) if n not in (x, y)]
    while True:
        best = max(
            (
                (wanted(a, b), -x, -y)
                for x, a in enumerate(rest)
                for y, b in enumerate(rest)
                if x < y
            ),
            default=(0, 0, 0),
        )
        if best[0] == 0:
            break
        x, y = -best[1], -best[2]
        a, b = rest[x], rest[y]
        rest = [it for n, it in enumerate(rest) if n not in (x, y)]
        up.append(f.mix(a, b))
    while rest:
        a = rest.pop(0)
        j = next((n for n, b in enumerate(rest) if f.comp(b) != f.comp(a)), 0)
        up.append(f.mix(a, rest.pop(j)))
    return up


def clone(f: Forest, item: Item) -> Item:
    """MTMD's CloneNode: a droplet of `item`'s composition, from U or rebuilt (Cases 1-3)."""
    if item[0] == "fluid":
        return item
    got = f.take(f.comps[item[1]])
    if got is not None:
        return got
    a, b = f.nodes[item[1]]
    return f.mix(clone(f, a), clone(f, b))


def mtmd(instance: Instance) -> PaperGraph:
    targets = instance.targets
    volumes = {t.droplet_volume for t in targets}
    if len(volumes) != 1:
        raise NotApplicable("targets of different droplet volumes (the forest's droplets are one)")
    v = volumes.pop()
    if 2 * v > instance.chip.max_mix_volume:
        raise NotApplicable(f"a {v}:{v} mix exceeds M = {instance.chip.max_mix_volume}")
    ratios = [_dyadic(instance, t.id) for t in targets]
    f, roots = mtr(ratios, instance.num_fluids)
    delivered: list[int] = []
    for t, root in zip(targets, roots, strict=True):
        if root[0] != "node":
            raise NotApplicable(f"target {t.id} is a pure fluid")
        for n in range(t.demand):
            item = root if n == 0 else clone(f, root)
            if item[0] != "node":
                raise AssertionError("a clone of a mix is a mix")
            delivered.append(item[1])
    return graph(f, delivered, v)


def graph(f: Forest, delivered: list[int], v: int) -> PaperGraph:
    nodes = tuple(
        tuple(In(v, fluid=x) if kind == "fluid" else In(v, node=x) for kind, x in pair)
        for pair in f.nodes
    )
    return PaperGraph(f.num_fluids, nodes, tuple(delivered))


def paper_counts(f: Forest, roots: list[Item]) -> tuple[int, int]:
    """(Tms, W) as the paper counts them: W leaves out each root's second droplet."""
    root_ids = {i for kind, i in roots if kind == "node"}
    waste = sum(len(ids) for ids in f.spare.values()) - sum(
        1 for ids in f.spare.values() for i in ids if i in root_ids
    )
    return len(f.nodes), waste


def _dyadic(instance: Instance, target: int) -> tuple[int, ...]:
    t = instance.targets[target]
    c = t.composition
    if c.den & (c.den - 1) == 0:
        return c.num
    if t.tolerance.num == 0:
        raise NotApplicable(f"target {target}: ratio sum {c.den} is not a power of two")
    limit = instance.chip.max_denominator
    n = limit.bit_length() - 1
    for depth in range(1, n + 1):
        best = _nearest(c, 1 << depth, t.tolerance)
        if best is not None:
            return best
    raise NotApplicable(f"target {target}: no k / 2^n within tolerance")


def _nearest(c: Composition, den: int, tol: object) -> tuple[int, ...] | None:
    """A ratio over `den` within `tol` of c on every fluid (largest-remainder rounding)."""
    from meda.core.composition import Rational, linf_distance

    raw = [x * den for x in c.num]
    base = [r // c.den for r in raw]
    order = sorted(range(len(raw)), key=lambda j: -(raw[j] - base[j] * c.den))
    for j in order[: den - sum(base)]:
        base[j] += 1
    cand = Composition.from_ints(tuple(base), den)
    assert isinstance(tol, Rational)
    return tuple(base) if linf_distance(cand, c) <= tol else None
