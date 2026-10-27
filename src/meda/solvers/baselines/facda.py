"""DFM and FacDA: factorisation-based MEDA dilution (Kundu et al., TCAD 41(3), 2022; SPEC 10.4).

Implemented from the paper (sections III-B to III-D), with the choices it leaves open written in
`notes/lit_bench/selection.md`:

- **FacDA-Transformation** (Alg. 1). A ratio x:y whose sum N has no factor set within [2, M] is
  moved to the next N' > N whose rounded ratio x' = round(x N' / N) is within the tolerance of
  the *original* target and whose N' factorises (the paper recurses on the moved ratio and
  measures each step against it; measuring against the original keeps the delivered droplet
  within tolerance). An exact target with no factor set is not applicable.
- **Feasible mixer set**: a least-cardinality factor set of N (all factors in [2, M]). Where
  several tie, the one whose DFM tree draws the least sample, then wastes least.
- **DFM** (III-C): the largest factor is the root mixer; the other positions, from the bottom
  up, take the factor that needs the fewest sample units there (ties: the larger factor). x and
  y written in that mixed radix give each node's sample and buffer units; every node passes one
  unit to its parent. Fig. 7: 71:49 at M = 6 is {5, 4, 6}, nm 3, ns 6, nw 7.
- **FacDA** (III-D): the same skeleton, where any node may pass volume to any ancestor (the
  constraints of the paper), choosing the fewest sample units. The paper solves this with an SMT
  solver; here an exhaustive search over the same integer variables gives the same minimum
  (ties, which the paper leaves to the solver: least waste, then the first found). Fig. 8(e):
  71:49 at M = 6 draws 4 sample units and wastes 3. A search above `SEARCH_BUDGET` nodes is not
  applicable rather than cut short.
"""

from __future__ import annotations

from math import prod

from meda.core.composition import Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import (
    In,
    NotApplicable,
    PaperGraph,
    PaperSolver,
    sample_and_buffer,
    single_target,
)

SEARCH_BUDGET = 3_000_000


def factor_sets(n: int, lo: int, hi: int) -> list[tuple[int, ...]]:
    """Every multiset of factors in [lo, hi] whose product is n (non-increasing tuples)."""
    out: list[tuple[int, ...]] = []

    def walk(rest: int, top: int, acc: tuple[int, ...]) -> None:
        if rest == 1:
            if acc:
                out.append(acc)
            return
        for f in range(min(top, rest), lo - 1, -1):
            if rest % f == 0:
                walk(rest // f, f, (*acc, f))

    walk(n, hi, ())
    return out


def _round_half_up(num: int, den: int) -> int:
    return (2 * num + den) // (2 * den)


def transform(instance: Instance) -> tuple[int, int, list[tuple[int, ...]]]:
    """FacDA-Transformation of the instance's target: (x', y', least-cardinality factor sets)."""
    single_target(instance)
    s, _ = sample_and_buffer(instance)
    target, chip = instance.targets[0], instance.chip
    return transform_ratio(
        target.composition.num[s],
        target.composition.den,
        target.tolerance,
        chip.min_mix_volume,
        chip.max_mix_volume,
        chip.max_denominator,
    )


def transform_ratio(
    x: int, n: int, tolerance: Rational, lo: int, hi: int, max_den: int
) -> tuple[int, int, list[tuple[int, ...]]]:
    """Alg. 1 on the ratio x : (n - x) as given (the paper does not reduce it first)."""
    sets = factor_sets(n, lo, hi)
    if sets:
        return x, n - x, _least(sets)
    if tolerance <= 0:
        raise NotApplicable(f"exact target {x}/{n}: {n} has no factors within [{lo}, {hi}]")
    for n2 in range(n + 1, max_den + 1):
        x2 = _round_half_up(x * n2, n)
        gap = Rational.of(x2, n2) - Rational.of(x, n)
        if not 0 < x2 < n2 or max(gap, -gap) > tolerance:
            continue
        sets = factor_sets(n2, lo, hi)
        if sets:
            return x2, n2 - x2, _least(sets)
    raise NotApplicable(
        f"no ratio within {tolerance} of {x}/{n} has a sum up to max_denominator {max_den} "
        f"that factorises within [{lo}, {hi}]"
    )


def _least(sets: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    d = min(len(f) for f in sets)
    return [f for f in sets if len(f) == d]


def dfm_order(x: int, factors: tuple[int, ...]) -> tuple[int, ...]:
    """DFM's ordered mixer set, bottom (position 0) to root: the largest factor at the root, the
    others chosen bottom-up by the fewest sample units at each position (ties: larger)."""
    rest = sorted(factors, reverse=True)
    root = rest.pop(0)
    order: list[int] = []
    q = x
    while rest:
        f = min(rest, key=lambda f: (q % f, -f))
        rest.remove(f)
        order.append(f)
        q //= f
    return (*order, root)


def _digits(n: int, radices: tuple[int, ...]) -> list[int]:
    out = []
    for f in radices:
        out.append(n % f)
        n //= f
    if n:
        raise ValueError(f"{n} left over after the mixed radix {radices}")
    return out


def dfm_graph(x: int, y: int, order: tuple[int, ...], sample: int, buffer: int) -> PaperGraph:
    """DFM's dilution tree for x:y on the ordered mixer set (bottom to root)."""
    xs, ys = _digits(x, order), _digits(y, order)
    nodes: list[tuple[In, ...]] = []
    for i, f in enumerate(order):
        node = [In(xs[i], fluid=sample), In(ys[i], fluid=buffer)]
        if nodes:
            node.append(In(1, node=len(nodes) - 1))
        if sum(u.units for u in node) == 0:
            continue
        if sum(u.units for u in node) != f:
            raise NotApplicable(f"{x}:{y} on {order}: position {i} holds no full mix of {f}")
        nodes.append(tuple(u for u in node if u.units))
    return PaperGraph(2, tuple(nodes))


def _dfm_choice(instance: Instance) -> tuple[PaperGraph, tuple[int, ...], int, int]:
    s, b = sample_and_buffer(instance)
    x, y, sets = transform(instance)
    best: tuple[tuple[int, int], PaperGraph, tuple[int, ...]] | None = None
    for factors in sets:
        order = dfm_order(x, factors)
        graph = dfm_graph(x, y, order, s, b)
        counts = graph.counts(order[-1])
        drawn = counts["drawn"]
        assert isinstance(drawn, tuple) and isinstance(counts["nw"], int)
        key = (drawn[s], counts["nw"])
        if best is None or key < best[0]:
            best = (key, graph, order)
    assert best is not None
    return best[1], best[2], x, y


def dfm(instance: Instance) -> PaperGraph:
    return _dfm_choice(instance)[0]


def facda(instance: Instance) -> PaperGraph:
    """FacDA's dilution graph: DFM's skeleton with sharing to any ancestor, fewest sample."""
    _, order, x, _ = _dfm_choice(instance)
    s, b = sample_and_buffer(instance)
    return facda_graph(x, order, s, b)


def facda_graph(x: int, order: tuple[int, ...], sample: int, buffer: int) -> PaperGraph:
    """The minimum-sample assignment of FacDA's variables on the skeleton `order` (III-D):
    node j takes s_j sample, b_j buffer and w_pj units of each lower node p, with
    s_j + b_j + sum_p w_pj = f_j, each term at most f_j - 1, and sum_j w_pj <= f_p."""
    d = len(order)
    p_after = [prod(order[: j + 1]) for j in range(d)]  # P_j: ratio-sum at node j
    between = [[prod(order[p + 1 : j]) for j in range(d)] for p in range(d)]
    best: list[object] = [None]  # (sample, waste, nodes)
    budget = [SEARCH_BUDGET]
    X = [0] * d  # sample numerator of each node over P_j
    cap = list(order)  # volume a node can still pass up
    nodes: list[tuple[In, ...]] = []

    def finish(total: int) -> None:
        graph = PaperGraph(2, tuple(nodes))
        waste = graph.counts(order[-1])["nw"]
        assert isinstance(waste, int)
        key = (total, waste)
        if best[0] is None or key < best[0][:2]:  # type: ignore[index]
            best[0] = (total, waste, tuple(nodes))

    def shares(j: int, p: int, room: int) -> list[list[int]]:
        if p == j:
            return [[]]
        out = []
        for w in range(min(cap[p], room, order[j] - 1), -1, -1):
            out += [[w, *rest] for rest in shares(j, p + 1, room - w)]
        return out

    def walk(j: int, total: int) -> None:
        budget[0] -= 1
        if budget[0] < 0:
            raise NotApplicable(f"FacDA's search exceeded {SEARCH_BUDGET} nodes on {order}")
        if best[0] is not None and total > best[0][0]:  # type: ignore[index]
            return
        f = order[j]
        for ws in shares(j, 0, f):
            contrib = sum(X[p] * w * between[p][j] for p, w in enumerate(ws))
            below = p_after[j - 1] if j else 1
            if j == d - 1:
                rest = x - contrib
                if rest % below:
                    continue
                choices = [rest // below]
            else:
                choices = list(range(f))
            for sj in choices:
                bj = f - sj - sum(ws)
                if not (0 <= sj <= f - 1 and 0 <= bj <= f - 1):
                    continue
                if best[0] is not None and total + sj > best[0][0]:  # type: ignore[index]
                    break
                Xj = sj * below + contrib
                if j < d - 1 and not 0 < Xj < p_after[j]:
                    continue  # a pure node is no mix
                node = [In(sj, fluid=sample), In(bj, fluid=buffer)]
                node += [In(w, node=p) for p, w in enumerate(ws)]
                X[j] = Xj
                for p, w in enumerate(ws):
                    cap[p] -= w
                nodes.append(tuple(u for u in node if u.units))
                if j == d - 1:
                    finish(total + sj)
                else:
                    walk(j + 1, total + sj)
                nodes.pop()
                for p, w in enumerate(ws):
                    cap[p] += w

    walk(0, 0)
    if best[0] is None:
        raise NotApplicable(f"no FacDA assignment on {order} reaches {x}/{p_after[-1]}")
    return PaperGraph(2, best[0][2])  # type: ignore[index]


class FacDASolver(PaperSolver):
    name = "facda"

    def __init__(self, weights: CostWeights) -> None:
        super().__init__("facda", facda, weights)
