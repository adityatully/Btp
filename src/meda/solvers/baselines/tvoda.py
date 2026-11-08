"""TVODA: target-volume-oriented dilution on MEDA (Kundu et al., "Mixing Models as Integer
Factorization", IEEE TCAD 41(3), 2022, Section IV; SPEC 10.4).

A target x : y (sum N) is wanted in Vt units, more than one mix makes. TVODA builds a forest of
skeleton trees (chains of mixers whose sizes multiply to N) whose roots together hold Vt, and
solves, with an SMT solver minimising the sample and buffer units, for the units of sample,
buffer and shared fluid at every node: inside one tree a node may pass fluid to any shallower
node (FacDA's edges), between trees from a node of ratio-sum P to a node whose child has the
same ratio-sum (the paper's restricted rule, P^n_{j+1} = P^m_i). Fig. 9(b): 71 : 49, M 6, Vt 13
is the forest [4, 5, 6], [4, 5, 6], [4, 5, 3, 2] (bottom up) with nm 9, ns 9, nw 6.

Our reading of what the paper leaves open (`notes/lit_bench/selection.md`):
- The forest (its ordering is in the paper's supplementary material, not on disk) follows
  Fig. 9: the least-cardinality factor set of N, ascending from the bottom (root the largest),
  as many copies as the root's volume fits whole droplets, then one tree whose root is the
  smallest factor of N that holds the remaining droplets, below it the longest bottom part of
  the main chain that divides what is left, then the least-cardinality factors of the rest.
- The SMT model is MTWM's ILP (`mtwm.solve`) with TVODA's edge rule and objective: least
  sample plus buffer units, then waste, then mixes; the same node cap applies.
- Demand d droplets of v units is Vt = d v, each droplet from one root. Exact targets only (the
  ratio-sum must factor within M; no transformation is given for volumes).
"""

from __future__ import annotations

from math import prod

from meda.core.problem import Instance
from meda.solvers.baselines.facda import factor_sets
from meda.solvers.baselines.mtwm import Node, solve
from meda.solvers.baselines.paper_graph import (
    NotApplicable,
    PaperGraph,
    sample_and_buffer,
)


def chain(m: int, bottom_up: list[int]) -> list[Node]:
    """A skeleton chain, deepest node first, root last (level 0)."""
    out, p = [], 1
    for depth, f in enumerate(bottom_up):
        p *= f
        out.append(Node(m, len(bottom_up) - 1 - depth, f, p))
    return out


def forest(n: int, lo: int, hi: int, droplets: int, v: int) -> list[list[int]]:
    """TVODA's skeleton trees (bottom-up factor lists) whose roots hold `droplets` of v."""
    sets = factor_sets(n, lo, hi)
    if not sets:
        raise NotApplicable(f"ratio-sum {n} has no factors within [{lo}, {hi}]")
    first: tuple[int, ...] = min(sets, key=_shortest)
    main = sorted(first)
    per = main[-1] // v
    if per < 1:
        raise NotApplicable(f"a droplet of {v} units exceeds the root mixer {main[-1]}")
    trees = [main] * (droplets // per)
    rest = droplets % per
    if rest:
        roots = [g for g in range(max(rest * v, lo), hi + 1) if n % g == 0]
        for g in roots:
            below = n // g
            k = len(main) - 1
            while k and below % prod(main[:k]):
                k -= 1
            left = below // prod(main[:k])
            tail: list[tuple[int, ...]] = factor_sets(left, lo, hi) if left > 1 else [()]
            if tail:
                least: tuple[int, ...] = min(tail, key=_shortest)
                trees.append(main[:k] + sorted(least) + [g])
                break
        else:
            raise NotApplicable(f"no root of {n}'s factors holds {rest} more droplets")
    return trees


def _shortest(f: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    return len(f), f


def tvoda_rule(src: Node, dst: Node) -> bool:
    if src.level == 0:
        return False
    if src.m == dst.m:
        return src.level > dst.level
    return dst.p // dst.f == src.p


def tvoda(instance: Instance) -> PaperGraph:
    if len(instance.targets) != 1:
        raise NotApplicable("TVODA makes one target")
    sample_and_buffer(instance)
    t, chip = instance.targets[0], instance.chip
    n = t.composition.den
    trees = forest(n, chip.min_mix_volume, chip.max_mix_volume, t.demand, t.droplet_volume)
    nodes = [nd for m, fs in enumerate(trees) for nd in chain(m, fs)]
    ratios = [t.composition.num] * len(trees)
    graph = solve(nodes, ratios, instance.num_fluids, instance.instance_id, tvoda_rule, True)
    roots: list[int] = []
    for r in graph.root_nodes():
        roots += [r] * (graph.volume(r) // t.droplet_volume)
    return PaperGraph(graph.num_fluids, graph.nodes, tuple(roots[: t.demand]))
