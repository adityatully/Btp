"""MTDA: multi-target dilution with an N-segment mixer (Kamal, Gupta, Singla, Tiwari, Kohli, Roy
& Bhattacharya, "Architectural Design of Flow-Based Microfluidic Biochips for Multi-Target
Dilution of Biochemical Fluids", ACM TODAES 25(3), 2020, Section 4; SPEC 10.4).

A target CF is c / N^z, written as z digits (0.d0 d1 ... d_{z-1}) in radix N. STDA (Algorithm 1)
builds its dilution tree top-down from the digits; every mix fills all N segments of the mixer:
- one digit (0.d0), a prime CF: d0 sample + (N - d0) buffer, (a : b);
- d0 <= N - 2, d1 != 0 and more digits after: d0 sample + 1 unit of the prime (0.d1) + 1 unit
  of (0.0 d2 ... d_{z-1}) + (N - d0 - 2) buffer, (a : 1 : 1 : b) (cases 7-9);
- otherwise: d0 sample + 1 unit of (0.d1 ... d_{z-1}) + (N - d0 - 1) buffer, (a : 1 : b) or
  (a : 1 : 0) (cases 2-6).
MTDA (Algorithm 3) runs STDA on every target in turn with one global sharing table
(SharingUtility, Algorithm 2): an intermediate CF made earlier, in this tree or another, gives
one unit to each later use, N uses per mix; when its N units are used up, a new mix makes it
again. A node is registered before its children are explored (preorder), and the IC2 subtree is
explored before IC1 (Algorithm 1, lines 8-9), which decides who shares with whom. A target is
never shared: it is used whole (Section 4.4.3). Fig. 5: 90/256 = (0.1122)_4 with mixer-4 is 5
mixes, (0.2)_4 shared by (0.022)_4 and (0.02)_4.

Our reading (`notes/lit_bench/coverage_audit.md`, paper log L-22):
- The mixer's N segments are N unit volumes of one MEDA mix, so N <= M. The paper's flow-based
  chip, storage architecture and scheduling (Sections 5-7) are not part of the plan.
- One accuracy level z for all targets, as in the paper's problem (Section 4.2). A tolerant
  target is moved to the nearest c / N^z within its tolerance, z the least with N^z within the
  chip's denominator limit; an exact target needs its CF to be c / N^z.
- N: the caller's `segments`, or else every N from 2 to M that applies, keeping the plan with
  the fewest mixes, then the least waste (the paper's objectives, Section 4.2; its MixerDesign
  of Section 5 weighs these with user-set weights).
- A target droplet of v units comes from a root mix of N units: floor(N / v) droplets per root
  mix, demand d from ceil(d / floor(N / v)) root mixes, each with its own tree (the paper:
  equal targets "are to be generated separately"); the rest of a root mix is waste. A target
  leaves the chip as soon as it is made (`early_outputs`): the paper's chip delivers it, and only
  intermediates are stored (Section 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, sample_and_buffer

Digits = tuple[int, ...]


def mtda(instance: Instance, segments: int | None = None) -> PaperGraph:
    """MTDA's dilution forest for a two-fluid instance (see the module docstring)."""
    s, b = sample_and_buffer(instance)
    m = instance.chip.max_mix_volume
    if segments is not None:
        if not 2 <= segments <= m:
            raise NotApplicable(f"mixer-{segments} needs 2 <= N <= M = {m}")
        return _forest(instance, segments, s, b)
    best: tuple[tuple[int, int], PaperGraph] | None = None
    why = []
    for n in range(2, m + 1):
        try:
            graph = _forest(instance, n, s, b)
        except NotApplicable as e:
            why.append(f"N={n}: {e}")
            continue
        delivered = sum(t.droplet_volume * t.demand for t in instance.targets)
        c = graph.counts(delivered)
        key = (int(c["nm"]), int(c["nw"]))  # type: ignore[call-overload]
        if best is None or key < best[0]:
            best = (key, graph)
    if best is None:
        raise NotApplicable("no mixer-N with 2 <= N <= M applies; " + "; ".join(why)[:150])
    return best[1]


def digits_of(instance: Instance, n: int, s: int) -> tuple[int, list[Digits]]:
    """(z, each target's z radix-n digits): the least z with n^z within the chip's denominator
    limit that puts every target's nearest c / n^z within its tolerance and strictly between
    0 and 1 (CFapprox, Algorithm 5, raises z until the target is met)."""
    top, den = 0, n
    while den <= instance.chip.max_denominator:
        top, den = top + 1, den * n
    for z in range(1, top + 1):
        den = n**z
        out: list[Digits] = []
        for t in instance.targets:
            c = t.composition.share(s)
            x = (2 * c.num * den + c.den) // (2 * c.den)  # nearest, ties up
            diff = Rational.of(x, den) - c
            if max(diff, -diff) > t.tolerance or not 0 < x < den:  # a pure fluid: next z
                break
            out.append(tuple((x // n ** (z - 1 - k)) % n for k in range(z)))
        else:
            return z, out
    raise NotApplicable(f"no c / {n}^z within tolerance with {n}^z <= max_denominator")


@dataclass
class _Node:
    sample: int
    buffer: int
    takes: list[int] = field(default_factory=list)  # one unit of each


class _Forest:
    """Algorithms 1-3 over one global sharing table."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.nodes: list[_Node] = []
        self.shared: dict[Digits, tuple[int, int]] = {}  # CF -> (current node, units used)

    def stda(self, digits: Digits, root: bool) -> int:
        """The node that gives one unit of (0.digits)_N (a root: the target's own mix)."""
        if not root:
            here = self.shared.get(digits)
            if here is not None and here[1] % self.n:  # SharingUtility: a unit is left
                self.shared[digits] = (here[0], here[1] + 1)
                return here[0]
        i = len(self.nodes)
        self.nodes.append(_Node(0, 0))
        if not root:  # registered before its children are explored (preorder)
            used = self.shared[digits][1] if digits in self.shared else 0
            self.shared[digits] = (i, used + 1)
        d0, rest = digits[0], digits[1:]
        n = self.n
        if not rest:  # prime CF, case 1
            self.nodes[i] = _Node(d0, n - d0)
        elif n - d0 > 1 and rest[0] != 0 and len(rest) > 1:  # (a : 1 : 1 : b), cases 7-9
            ic2 = self.stda(_strip((0, *rest[1:])), False)
            ic1 = self.stda((rest[0],), False)
            self.nodes[i] = _Node(d0, n - d0 - 2, [ic1, ic2])
        else:  # (a : 1 : b) or (a : 1 : 0), cases 2-6
            child = self.stda(rest, False)
            self.nodes[i] = _Node(d0, n - d0 - 1, [child])
        return i


def _strip(d: Digits) -> Digits:
    end = len(d)
    while end and d[end - 1] == 0:
        end -= 1
    return d[:end]


def _forest(instance: Instance, n: int, s: int, b: int) -> PaperGraph:
    _, digits = digits_of(instance, n, s)
    f = _Forest(n)
    roots: list[int] = []
    for t, d in zip(instance.targets, digits, strict=True):
        per = n // t.droplet_volume
        if per == 0:
            raise NotApplicable(f"target {t.id}: a {t.droplet_volume}-unit droplet > mixer-{n}")
        left = t.demand
        while left:
            q = min(per, left)
            roots += [f.stda(_strip(d), True)] * q
            left -= q
    # children before parents: depth-first post-order from the roots
    order: list[int] = []
    state = [0] * len(f.nodes)  # 0 new, 1 open, 2 done

    def visit(i: int) -> None:
        if state[i] == 2:
            return
        if state[i] == 1:
            raise ValueError(f"MTDA forest has a cycle through node {i}")
        state[i] = 1
        for c in f.nodes[i].takes:
            visit(c)
        state[i] = 2
        order.append(i)

    for r in roots:
        visit(r)
    pos = {old: new for new, old in enumerate(order)}
    nodes = []
    for old in order:
        node = f.nodes[old]
        ins = [In(node.sample, fluid=s)] if node.sample else []
        units: dict[int, int] = {}
        for c in node.takes:
            units[pos[c]] = units.get(pos[c], 0) + 1
        ins += [In(u, node=c) for c, u in units.items()]
        if node.buffer:
            ins.append(In(node.buffer, fluid=b))
        nodes.append(tuple(ins))
    return PaperGraph(
        instance.num_fluids, tuple(nodes), tuple(pos[r] for r in roots), early_outputs=True
    )
