"""STWM and MTWM: waste-minimising mixing on MEDA over DFMM skeletons (Kundu & Roy, "Multi-target
Fluid Mixing in MEDA Biochips: Theory and an Attempt toward Waste Minimization", ACM TODAES 28(6),
2023, Sections 6-7; STWM first in DSD 2022; SPEC 10.4).

Each target's DFMM tree (`dfmm.dfmm`) gives a skeleton: its mix nodes, each with its level, its
mixer size f and its ratio-sum P (the product of the f's on the longest path from the node down
to a leaf). Over the skeletons the paper solves, with an SMT solver, for each node's reagent units
r^t, the units w it takes from other nodes and its ratio R^t (an integer composition over P):

- (1)/(8) P r^t + sum_j (P / P_j) w_j R_j^t == f R^t (the concentration formula, times f);
- (2)/(9) sum_t R^t == P;  (4)/(11) the root's R is the target;
- (5)/(12) a node's inputs total f, or 0 (the node is pruned);
- (6)/(13) 0 <= w <= f - 1;  (7)/(14) 0 <= r^t <= f - 1;
- an edge from v_j to v is allowed when (P / f) mod P_j == 0, and, inside one skeleton, when
  v_j is on a deeper level (STWM's multi-level sharing); between skeletons (MTWM) the modulo
  rule alone. The rule makes P grow along every edge, so the graph has no cycle.

Objective: least waste nw (the paper's stated aim), where every non-root node's volume not passed
on is waste. Fig. 15 (2:11:5, 12:5:1, 5:6:14, M 5): nw 8, nr 20, nm 8; Table 2 (protein
electrophoresis, M 7).

Our reading of what the paper leaves open (`notes/lit_bench/selection.md`):
- The clauses state no objective (an SMT model is a feasibility check). We minimise nw, then the
  reagent units nr, then the mixes nm, exactly. The product w R is made linear by writing w in
  one-hot form (w <= f - 1 is small) and the model is an ILP solved by HiGHS
  (`scipy.optimize.milp`, gap 0); a solve over `TIME_LIMIT` is not applicable. The paper caps the
  sharing edges on large cases without saying how; we never cap.
- Two rules the clauses leave implicit and a mixing graph needs: a pruned node has no ratio
  (sum R = P y, y = 0 when pruned) and passes nothing on, and a node passes on at most its volume.
- Roots pass nothing on (a root holds the target; the paper's nw counts no root).
- One droplet per target (demand 1), each built exactly (no ratio transformation is given for a
  ratio whose DFMM factors do not exist; `dfmm` raises).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from meda.core.problem import Instance
from meda.solvers.baselines.dfmm import dfmm
from meda.solvers.baselines.nfsp import _Model
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, single_target

NODE_LIMIT = 2_000  # branch-and-bound nodes: past it, the best graph found (deterministic)
TIME_LIMIT = 300.0  # seconds: a safety stop, past it not applicable


@dataclass(frozen=True, slots=True)
class Node:
    """A skeleton mix node: target (tree) `m`, `level` (root 0), mixer size `f`, ratio-sum `p`."""

    m: int
    level: int
    f: int
    p: int


def skeleton(graph: PaperGraph, m: int) -> list[Node]:
    """The nodes of a DFMM tree with their levels and ratio-sums (root last, as in `graph`)."""
    n = len(graph.nodes)
    level = [0] * n
    for i in range(n - 1, -1, -1):
        for x in graph.nodes[i]:
            if x.node is not None:
                level[x.node] = level[i] + 1
    p = [0] * n
    for i, node in enumerate(graph.nodes):  # children come first
        below = [p[x.node] for x in node if x.node is not None]
        p[i] = graph.volume(i) * max(below, default=1)
    return [Node(m, level[i], graph.volume(i), p[i]) for i in range(n)]


def allowed(src: Node, dst: Node) -> bool:
    """The paper's sharing rule: (P / f) mod P_src == 0, and deeper within one skeleton."""
    if src.level == 0:  # roots pass nothing on
        return False
    if src.m == dst.m and src.level <= dst.level:
        return False
    return (dst.p // dst.f) % src.p == 0


def mtwm(instance: Instance) -> PaperGraph:
    targets = instance.targets
    if any(t.demand != 1 for t in targets):
        raise NotApplicable("MTWM makes one droplet of each target (demand 1)")
    nodes: list[Node] = []
    for m, t in enumerate(targets):
        one = Instance(
            instance.instance_id, instance.fluids, (replace(t, id=0),), instance.chip, 10**9
        )
        nodes += skeleton(dfmm(one), m)
    ratios = [t.composition.num for t in targets]
    return solve(nodes, ratios, instance.num_fluids, instance.instance_id)


def stwm(instance: Instance) -> PaperGraph:
    single_target(instance)
    return mtwm(instance)


def solve(nodes: list[Node], ratios: list[tuple[int, ...]], fluids: int, what: str) -> PaperGraph:
    """The paper's model over `nodes` as an ILP; the least-waste mixing graph it allows."""
    lp = _Model()
    n = len(nodes)
    root = [nd.level == 0 for nd in nodes]
    y = [lp.var() for _ in nodes]
    r = [[lp.var() for _ in range(fluids)] for _ in nodes]
    big_r = [[lp.var() for _ in range(fluids)] for _ in nodes]
    edges = [(s, d) for d in range(n) for s in range(n) if s != d and allowed(nodes[s], nodes[d])]
    # one-hot w: b[e][k - 1] == 1 iff edge e carries k units, k in 1..f_dst - 1
    b = [[lp.var() for _ in range(nodes[d].f - 1)] for _, d in edges]
    z = [[[lp.var() for _ in range(nodes[d].f - 1)] for _ in range(fluids)] for _, d in edges]
    inflow: list[dict[int, float]] = [{} for _ in nodes]
    out: list[dict[int, float]] = [{} for _ in nodes]
    ratio: list[list[dict[int, float]]] = [[{} for _ in range(fluids)] for _ in nodes]

    def bound(j: int, lo: int, hi: int) -> None:
        lp.row({j: 1.0}, lo, hi)

    for i, nd in enumerate(nodes):
        bound(y[i], 1 if root[i] else 0, 1)
        for t in range(fluids):
            lp.row({r[i][t]: 1.0, y[i]: -(nd.f - 1)}, -np.inf, 0)  # r <= (f - 1) y
            bound(r[i][t], 0, nd.f - 1)
            bound(big_r[i][t], 0, nd.p)
            inflow[i][r[i][t]] = 1.0
            ratio[i][t] = {r[i][t]: float(nd.p), big_r[i][t]: -float(nd.f)}
        lp.row({**{big_r[i][t]: 1.0 for t in range(fluids)}, y[i]: -float(nd.p)}, 0, 0)
        if root[i]:
            tgt = ratios[nd.m]
            scale, rem = divmod(nd.p, sum(tgt))
            assert rem == 0, "a DFMM root's ratio-sum is the target's"
            for t in range(fluids):
                bound(big_r[i][t], tgt[t] * scale, tgt[t] * scale)
    for e, (s, d) in enumerate(edges):
        ps, coef = nodes[s].p, nodes[d].p // nodes[s].p
        lp.row({x: 1.0 for x in b[e]}, 0, 1)
        for k, bk in enumerate(b[e], start=1):
            bound(bk, 0, 1)
            inflow[d][bk] = float(k)
            out[s][bk] = float(k)
            for t in range(fluids):
                zk = z[e][t][k - 1]
                bound(zk, 0, ps)
                lp.row({zk: 1.0, bk: -float(ps)}, -np.inf, 0)  # z <= P_s b
                lp.row({zk: 1.0, big_r[s][t]: -1.0}, -np.inf, 0)  # z <= R_s
                lp.row({zk: 1.0, big_r[s][t]: -1.0, bk: -float(ps)}, -float(ps), np.inf)
                ratio[d][t][zk] = float(coef * k)
    for i, nd in enumerate(nodes):
        lp.row({**inflow[i], y[i]: -float(nd.f)}, 0, 0)
        for t in range(fluids):
            lp.row(ratio[i][t], 0, 0)
        if out[i]:
            lp.row({**out[i], y[i]: -float(nd.f)}, -np.inf, 0)
    # lexicographic: waste, then reagent units, then mixes
    w_nm = 1.0
    w_nr = float(n + 1)
    w_nw = w_nr * float(sum(nd.f for nd in nodes) + 1)
    objective: dict[int, float] = {}
    for i, nd in enumerate(nodes):
        objective[y[i]] = w_nm + (0.0 if root[i] else w_nw * nd.f)
        for t in range(fluids):
            objective[r[i][t]] = w_nr
        for bk, units in out[i].items():
            objective[bk] = objective.get(bk, 0.0) - w_nw * units
    res = lp.run(
        objective, {"node_limit": NODE_LIMIT, "time_limit": TIME_LIMIT, "mip_rel_gap": 0.0}
    )
    if res.x is None:
        raise NotApplicable(f"MTWM on {what}: no graph within the solver's limits ({res.message})")
    if res.status == 1 and "node" not in res.message.lower():
        raise NotApplicable(f"MTWM on {what}: the ILP hit {TIME_LIMIT:g} s ({res.message})")
    x = np.round(res.x).astype(int)
    return _graph(nodes, edges, b, r, y, x, fluids)


def _graph(
    nodes: list[Node],
    edges: list[tuple[int, int]],
    b: list[list[int]],
    r: list[list[int]],
    y: list[int],
    x: np.ndarray,
    fluids: int,
) -> PaperGraph:
    keep = [i for i in range(len(nodes)) if x[y[i]]]
    keep.sort(key=lambda i: (nodes[i].p, nodes[i].m, -nodes[i].level))  # edges raise P
    at = {i: k for k, i in enumerate(keep)}
    ins: dict[int, list[In]] = {i: [] for i in keep}
    for i in keep:
        ins[i] += [In(int(x[r[i][t]]), fluid=t) for t in range(fluids) if x[r[i][t]]]
    for e, (s, d) in enumerate(edges):
        k = sum(kk for kk, bk in enumerate(b[e], start=1) if x[bk])
        if k:
            ins[d].append(In(k, node=at[s]))
    roots = tuple(
        sorted((at[i] for i in keep if nodes[i].level == 0), key=lambda a: nodes[keep[a]].m)
    )
    return PaperGraph(fluids, tuple(tuple(ins[i]) for i in keep), roots)
