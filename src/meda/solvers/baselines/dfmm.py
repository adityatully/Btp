"""DFMM: division-by-factor mixing on MEDA (Kundu & Roy, DSD 2022, section IV; SPEC 10.4).

DFM extended to t >= 2 fluids. The ratio-sum S = r_1 + ... + r_t is split into the shortest
list of factors within [2, M], in descending order f_0 >= ... >= f_d, f_0 at the root. From the
deepest level up, each r_i is divided by the level's factor: the remainders are the units of
fluid i mixed at that level and the quotients go up a level. A level holds (its reagent units +
one unit from each node below) / f nodes, each a mix of f units passing one unit up. Fig. 5:
7:7:6 at M = 5 is L = [5, 4], two mixes of 4 under a root of 5, nr 11, nw 6, nm 3.

Choices the paper leaves open (`notes/lit_bench/selection.md`):
- Which units share a node is not stated (the totals nm, nr, nw do not depend on it): units are
  dealt to the level's nodes in fluid order, children last, and one unit is swapped between
  nodes where that would leave a node holding a single fluid (no mix).
- Ties between shortest factor lists: the fewest reagent units, then the least waste.
- The paper takes the target ratio as given; a ratio-sum with a prime factor above M is not
  applicable (no transformation is given for mixtures).
"""

from __future__ import annotations

from meda.core.problem import Instance
from meda.solvers.baselines.facda import factor_sets
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, single_target


def dfmm(instance: Instance) -> PaperGraph:
    single_target(instance)
    target, chip = instance.targets[0], instance.chip
    ratio, total = target.composition.num, target.composition.den
    sets = factor_sets(total, chip.min_mix_volume, chip.max_mix_volume)
    if not sets:
        raise NotApplicable(
            f"{target.composition}: ratio-sum {total} has no factors within "
            f"[{chip.min_mix_volume}, {chip.max_mix_volume}]"
        )
    shortest = min(len(f) for f in sets)
    best: tuple[tuple[int, int], PaperGraph] | None = None
    for factors in (f for f in sets if len(f) == shortest):
        graph = dfmm_graph(ratio, factors)
        counts = graph.counts(factors[0])
        drawn, nw = counts["drawn"], counts["nw"]
        assert isinstance(drawn, tuple) and isinstance(nw, int)
        key = (sum(drawn), nw)
        if best is None or key < best[0]:
            best = (key, graph)
    assert best is not None
    return best[1]


def dfmm_graph(ratio: tuple[int, ...], factors: tuple[int, ...]) -> PaperGraph:
    """DFMM's mixing tree for `ratio` on `factors` (descending, root first)."""
    quotients = list(ratio)
    nodes: list[tuple[In, ...]] = []
    below: list[int] = []  # node indices of the level underneath
    for level, f in reversed(list(enumerate(factors))):
        units: list[tuple[str, int]] = []
        for i, q in enumerate(quotients):
            units += [("fluid", i)] * (q % f)
            quotients[i] = q // f
        units += [("node", n) for n in below]
        if len(units) % f:
            raise NotApplicable(f"{ratio} on {factors}: level {level} holds {len(units)} units")
        groups = _no_single_fluid([units[k : k + f] for k in range(0, len(units), f)])
        below = []
        for group in groups:
            node = [In(group.count(("fluid", i)), fluid=i) for i in range(len(ratio))]
            node += [In(1, node=n) for kind, n in group if kind == "node"]
            below.append(len(nodes))
            nodes.append(tuple(u for u in node if u.units))
    if any(quotients) or len(below) != 1:
        raise NotApplicable(f"{ratio} on {factors} does not end in one root mix")
    return PaperGraph(len(ratio), tuple(nodes))


def _no_single_fluid(groups: list[list[tuple[str, int]]]) -> list[list[tuple[str, int]]]:
    """Swap one unit between nodes so that no node is a single fluid on its own."""
    for g in groups:
        if len(set(g)) > 1 or g[0][0] == "node":
            continue
        for other in groups:
            for k, unit in enumerate(other):
                if unit == g[0]:
                    continue
                rest = other[:k] + other[k + 1 :] + [g[0]]
                if len(set(rest)) > 1 or rest[0][0] == "node":
                    other[k], g[-1] = g[-1], unit
                    break
            else:
                continue
            break
    return groups
