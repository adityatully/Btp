"""NFSP: network-flow sample preparation, optimal within 1:1 dilution (Dinh, Yamashita and Ho,
"A Network-Flow-Based Optimal Sample Preparation Algorithm for Digital Microfluidic Biochips",
ASP-DAC 2014; SPEC 10.4).

At precision 2^-d, concentrations are numerators c in 0..2^d. Level 0 holds buffer (0) and
sample (2^d); level l + 1 holds every value a 1:1 dilution of two level-l values makes, and a
value may also pass down a level unchanged. A dilution of c_i with c_j takes one unit droplet of
each and gives two of (c_i + c_j) / 2: in the flow network its two input arcs carry equal flow
(the integer equal-flows constraint). Targets leave level d with their droplet counts, the rest
of level d is waste. The ILP minimises F = cost_s * sample + cost_b * buffer, so it is optimal
among all 1:1 dilution graphs at that precision, for any number of targets and droplets. Paper
check (its Fig. 2): d = 3, targets 3/8 x 4 and 5/8 x 6, cost_s 2, cost_b 1 -> sample 6, buffer
5, one waste droplet of 6/8, F = 17.

Our reading of what the paper leaves open (`notes/lit_bench/objectives.md`):
- Costs are the instance's fluid weights (the paper's general F; its `ours_S` and `ours_W` are
  the weights (1, 0) and (1, 1)). Ties, left to CPLEX in the paper, go to the fewest mixes and
  then the least waste in our weights (a second ILP with F held at its optimum).
- Precision: an exact target needs a power-of-2 denominator (else not applicable). A tolerant
  target is moved to the nearest c / 2^d within its tolerance (ties to less sample), with the
  smallest d that serves every target and 2^d at most the chip's `max_denominator`.
- MEDA: n dilutions of the same pair at one level are one n:n mix when 2n <= M (that is how a
  MEDA chip runs 1:1 dilutions in parallel); `half` = M // 2 bounds n. A target droplet of v
  units is taken from one mix, which our engine needs (two droplets of one composition cannot be
  merged): the ILP gets, per pair making a target value, mixes of a:a units (2a >= v) set aside
  for that target, each giving up to 2a // v of its droplets, and those units are held back
  from later dilutions. With v = 1 and M = 2 this is the paper's model unchanged.
- The ILP is HiGHS (`scipy.optimize.milp`, gap 0) instead of CPLEX; a solve over `TIME_LIMIT`
  is not applicable. Plans that overfill the chip's storage are not applicable (`emit`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import coo_array

from meda.core.composition import Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, sample_and_buffer

TIME_LIMIT = 300.0  # seconds per ILP solve


@dataclass(frozen=True, slots=True)
class Want:
    """`demand` droplets of `volume` units at numerator `c` over 2^d."""

    c: int
    volume: int
    demand: int


@dataclass
class Flow:
    """An optimal flow: units of sample and buffer, and the dilutions by (level, c_i, c_j)."""

    d: int
    sample: int
    buffer: int
    free: dict[tuple[int, int, int], int] = field(default_factory=dict)  # unit dilutions
    # (level, c_i, c_j, a, target) -> (a:a mixes set aside for the target, droplets delivered)
    kept: dict[tuple[int, int, int, int, int], tuple[int, int]] = field(default_factory=dict)

    def dilutions(self) -> int:
        """The paper's ND: unit 1:1 dilutions."""
        return sum(self.free.values()) + sum(a * n for (_, _, _, a, _), (n, _) in self.kept.items())


class _Model:
    def __init__(self) -> None:
        self.n = 0
        self.rows: list[dict[int, float]] = []
        self.lo: list[float] = []
        self.hi: list[float] = []

    def var(self) -> int:
        self.n += 1
        return self.n - 1

    def row(self, coef: dict[int, float], lo: float, hi: float) -> None:
        self.rows.append(coef)
        self.lo.append(lo)
        self.hi.append(hi)

    def solve(
        self, objective: dict[int, float], what: str, time_limit: float = TIME_LIMIT
    ) -> np.ndarray:
        res = self.run(objective, {"time_limit": time_limit, "mip_rel_gap": 0.0})
        if res.status == 1:
            raise NotApplicable(f"{what}: the ILP was not solved within {time_limit:g} s")
        if res.status != 0 or res.x is None:
            raise NotApplicable(f"{what}: no flow ({res.message})")
        return np.asarray(res.x)

    def run(self, objective: dict[int, float], options: dict[str, Any]) -> Any:
        """`scipy.optimize.milp` on the model, all variables integer; its raw result."""
        r, c, v = [], [], []
        for i, coef in enumerate(self.rows):
            for j, x in coef.items():
                r.append(i)
                c.append(j)
                v.append(x)
        a = coo_array((v, (r, c)), shape=(len(self.rows), self.n)).tocsr()
        cost = np.zeros(self.n)
        for j, x in objective.items():
            cost[j] = x
        return milp(
            cost,
            constraints=LinearConstraint(a, np.array(self.lo), np.array(self.hi)),
            integrality=np.ones(self.n),
            options=options,
        )


def flow(
    d: int,
    wants: list[Want],
    cost_s: float,
    cost_b: float,
    half: int,
    tie: tuple[float, float] | None = (1.0, 1.0),
) -> Flow:
    """The minimum-cost flow at precision 2^-d for `wants`, with mixes of at most `half`:`half`
    units. `tie` = (weight per mix, weight per waste unit) breaks ties in a second solve; None
    skips it (F and the fluid counts are then still optimal)."""
    D = 1 << d
    if not all(0 < w.c < D for w in wants):
        raise NotApplicable("a target is a pure fluid at this precision")
    if any(w.volume > 2 * half for w in wants):
        raise NotApplicable(f"a target droplet needs a mix of more than {2 * half} units")
    m = _Model()
    src_s, src_b = m.var(), m.var()
    inflow: dict[tuple[int, int], dict[int, float]] = {}
    outflow: dict[tuple[int, int], dict[int, float]] = {}

    def add(table: dict[tuple[int, int], dict[int, float]], key: tuple[int, int], j: int,
            x: float) -> None:  # fmt: skip
        coef = table.setdefault(key, {})
        coef[j] = coef.get(j, 0.0) + x

    add(inflow, (0, D), src_s, 1)
    add(inflow, (0, 0), src_b, 1)
    passes: dict[tuple[int, int], int] = {}
    free: dict[tuple[int, int, int], int] = {}
    kept: dict[tuple[int, int, int, int, int], tuple[int, int]] = {}
    targets_at: dict[int, list[int]] = {}
    for t, w in enumerate(wants):
        targets_at.setdefault(w.c, []).append(t)
    for level in range(d):
        step = D >> level
        values = range(0, D + 1, step)
        for c in values:
            j = passes[(level, c)] = m.var()
            add(outflow, (level, c), j, 1)
            add(inflow, (level + 1, c), j, 1)
        for i in values:
            for k in range(i + step, D + 1, step):
                made = (i + k) // 2
                j = free[(level, i, k)] = m.var()
                for src in (i, k):
                    add(outflow, (level, src), j, 1)
                add(inflow, (level + 1, made), j, 2)
                for t in targets_at.get(made, ()):
                    v = wants[t].volume
                    for a in range((v + 1) // 2, half + 1):
                        n, q = m.var(), m.var()
                        kept[(level, i, k, a, t)] = (n, q)
                        for src in (i, k):
                            add(outflow, (level, src), n, a)
                        add(inflow, (level + 1, made), n, 2 * a)
                        m.row({q: 1, n: -(2 * a // v)}, -np.inf, 0)
    waste = {}
    for c in range(D + 1):
        waste[c] = m.var()
        add(outflow, (d, c), waste[c], 1)
    for key in set(inflow) | set(outflow):
        coef = dict(inflow.get(key, {}))
        for j, x in outflow.get(key, {}).items():
            coef[j] = coef.get(j, 0.0) - x
        sink = sum(w.volume * w.demand for w in wants if key[0] == d and w.c == key[1])
        m.row(coef, sink, sink)
    for t, w in enumerate(wants):
        m.row({q: 1 for key, (_, q) in kept.items() if key[4] == t}, w.demand, w.demand)
    # Units set aside for a target leave level d as delivered: they pass every later level.
    for c, ts in targets_at.items():
        for level in range(1, d):
            held = {
                q: -wants[key[4]].volume
                for key, (_, q) in kept.items()
                if key[4] in ts and key[0] < level
            }
            if held:
                m.row({passes[(level, c)]: 1, **held}, 0, np.inf)
    fluid = {src_s: cost_s, src_b: cost_b}
    sol = m.solve(fluid, f"NFSP at d = {d}")
    if tie is not None:
        best = cost_s * round(sol[src_s]) + cost_b * round(sol[src_b])
        m.row(fluid, -np.inf, best + 1e-6)
        mixes = {}
        for j in free.values():
            nodes = m.var()  # free a:a mixes of this pair
            m.row({nodes: half, j: -1}, 0, np.inf)
            mixes[nodes] = tie[0]
        for n, _ in kept.values():
            mixes[n] = tie[0]
        for c in range(D + 1):
            mixes[waste[c]] = tie[1]
        sol = m.solve(mixes, f"NFSP ties at d = {d}")

    def val(j: int) -> int:
        return round(sol[j])

    return Flow(
        d,
        val(src_s),
        val(src_b),
        {key: val(j) for key, j in free.items() if val(j)},
        {key: (val(n), val(q)) for key, (n, q) in kept.items() if val(n)},
    )


def precision(instance: Instance) -> tuple[int, list[Want]]:
    """The smallest d that serves every target, and each target's numerator at 2^-d."""
    sample, _ = sample_and_buffer(instance)
    fracs = [(t.composition.num[sample], t.composition.den, t.tolerance) for t in instance.targets]
    limit = instance.chip.max_denominator
    d = 1
    while (1 << d) <= limit:
        cs = [_nearest(p, q, tol, 1 << d) for p, q, tol in fracs]
        if all(c is not None for c in cs):
            return d, [
                Want(c, t.droplet_volume, t.demand)
                for c, t in zip(cs, instance.targets, strict=True)
                if c is not None
            ]
        d += 1
    raise NotApplicable(
        f"no target set on a 1/2^d grid within tolerance, 2^d up to max_denominator {limit}"
    )


def _nearest(p: int, q: int, tol: Rational, D: int) -> int | None:
    """The numerator over D nearest to p / q within `tol`, strictly between 0 and D."""
    best = None
    for c in sorted({p * D // q, -(-p * D // q)}):
        gap = abs(c * q - p * D)  # |c / D - p / q| * D * q
        fits = 0 < c < D and gap * tol.den <= tol.num * D * q
        if fits and (best is None or gap < best[0]):
            best = (gap, c)
    return None if best is None else best[1]


def nfsp(instance: Instance, weights: CostWeights) -> PaperGraph:
    sample, buffer = sample_and_buffer(instance)
    d, wants = precision(instance)
    fluids = instance.fluids
    tie = (float(weights.w_mix), float(weights.w_waste))
    found = flow(
        d,
        wants,
        float(weights.fluid_weight(fluids[sample].kind)),
        float(weights.fluid_weight(fluids[buffer].kind)),
        instance.chip.max_mix_volume // 2,
        tie,
    )
    return graph_of(found, wants, sample, buffer, instance.chip.max_mix_volume // 2)


def graph_of(found: Flow, wants: list[Want], sample: int, buffer: int, half: int) -> PaperGraph:
    """The flow as mixes: per level and pair, the mixes set aside for targets, then the free
    dilutions in mixes of up to `half`:`half`; each draws on droplets of earlier levels (or a
    reservoir), oldest first. Set-aside units go to their targets, never to a later mix."""
    D = 1 << found.d
    nodes: list[tuple[In, ...]] = []
    pool: dict[int, list[list[int]]] = {}  # value -> [[node, units left], ...]
    outputs: list[list[int]] = [[] for _ in wants]

    def draw(c: int, units: int) -> list[In]:
        if c in (0, D):
            return [In(units, fluid=sample if c == D else buffer)]
        ins = []
        for entry in pool.get(c, []):
            take = min(units, entry[1])
            if take:
                ins.append(In(take, node=entry[0]))
                entry[1] -= take
                units -= take
        if units:
            raise ValueError(f"the flow draws {units} more units of {c}/{D} than it made")
        return ins

    for level in range(found.d):
        made: list[tuple[int, int, int]] = []  # (value, node, units for later levels)
        pairs = sorted({k[:3] for k in found.free} | {k[:3] for k in found.kept}, key=lambda k: k)
        for key in (k for k in pairs if k[0] == level):
            _, i, k = key
            c = (i + k) // 2
            for (lv, i2, k2, a, t), (n, q) in sorted(found.kept.items()):
                if (lv, i2, k2) != key:
                    continue
                each = 2 * a // wants[t].volume
                for _ in range(n):
                    nodes.append((*draw(i, a), *draw(k, a)))
                    give = min(each, q)
                    q -= give
                    outputs[t] += [len(nodes) - 1] * give
                    made.append((c, len(nodes) - 1, 2 * a - give * wants[t].volume))
            ops = found.free.get(key, 0)
            while ops:
                a = min(ops, half)
                ops -= a
                nodes.append((*draw(i, a), *draw(k, a)))
                made.append((c, len(nodes) - 1, 2 * a))
        for c, node, units in made:
            pool.setdefault(c, []).append([node, units])
    roots = tuple(r for out in outputs for r in out)
    return PaperGraph(2, tuple(nodes), roots)
