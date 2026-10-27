"""Generic sample preparation (Poddar, Fink, Haselmayr, Wille, TCAD 41(11), 2022; SPEC 10.4).

Alg. 1 instantiated for MEDA (dispense 1..M units, mixes of 2..M units, unequal mixing and
splitting, child droplets of 1..M-1 units). Top down from the target CF t/N: if sample and buffer
alone make it (step 3) the chain ends; otherwise the first CF of the estimated set E_cf (step 4),
then of the rest R_cf (step 5), from which some sample, buffer and v units of that CF make it
becomes the next target, and the next mix must make at least v units. Fig. 6(b): 7/12 at M = 4
is 6/12 = 1 S + 1 B, 2/12 = 2 B + 1 of 6/12, 7/12 = 1 S + 1 of 2/12: 3 mixes, 2 sample units.

Choices the paper leaves open (`notes/lit_bench/selection.md`):
- E_cf ("CFs selected from S1 = {N/2 .. 1} and S2 = {N/2 .. N-1} in a gap of ceil(sqrt N)"):
  every ceil(sqrt N)-th CF from N // 2 downwards, then upwards; for N = 12 that is {6, 2, 10},
  as in Fig. 6(b). R_cf is every other CF in ascending order.
- Loops run sample, then buffer, then child volume, each ascending; the first match is taken.
- Line 1 cites FacDA for writing C_t as x/(x+y): a tolerant target goes through
  FacDA-Transformation, and every mix is then exact on that grid (the paper's per-step check
  against the tolerance could add up past it).
"""

from __future__ import annotations

from math import isqrt

from meda.core.problem import Instance
from meda.solvers.baselines.facda import transform
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, sample_and_buffer


def estimated_cfs(n: int) -> list[int]:
    gap = isqrt(n - 1) + 1 if n > 1 else 1
    mid = n // 2
    down = list(range(mid, 0, -gap))
    up = [c for c in range(mid, n, gap) if c not in down]
    return down + up


def generic(instance: Instance) -> PaperGraph:
    s, b = sample_and_buffer(instance)
    x, y, _ = transform(instance)
    m = instance.chip.max_mix_volume
    lo = max(instance.chip.min_mix_volume, instance.targets[0].droplet_volume)
    return generic_graph(x, x + y, m, lo, s, b)


def generic_graph(x: int, n: int, m: int, lo: int, sample: int, buffer: int) -> PaperGraph:
    """The chain of mixes Alg. 1 finds for the CF x/n with mixes of `lo`..`m` units."""
    ecf = [c for c in estimated_cfs(n) if c != x]
    rcf = [c for c in range(1, n) if c != x and c not in ecf]
    steps: list[tuple[int, int, int, int]] = []  # (sample, buffer, child units, child CF)
    tr, least = x, lo
    while True:
        direct = next(
            (
                (sv, bv)
                for sv in range(1, m + 1)
                for bv in range(1, m + 1)
                if least <= sv + bv <= m and sv * n == tr * (sv + bv)
            ),
            None,
        )
        if direct is not None:
            break
        found = _gen_cf(tr, ecf, n, m, least) or _gen_cf(tr, rcf, n, m, least)
        if found is None:
            raise NotApplicable(f"Generic finds no mix for CF {tr}/{n} at M = {m}")
        steps.append(found)
        tr, least = found[3], max(2, found[2])
        (ecf if tr in ecf else rcf).remove(tr)
    nodes: list[tuple[In, ...]] = [(In(direct[0], fluid=sample), In(direct[1], fluid=buffer))]
    for sv, bv, v, _ in reversed(steps):
        node = (In(sv, fluid=sample), In(bv, fluid=buffer), In(v, node=len(nodes) - 1))
        nodes.append(tuple(u for u in node if u.units))
    return PaperGraph(2, tuple(nodes))


def _gen_cf(
    tr: int, cfs: list[int], n: int, m: int, least: int
) -> tuple[int, int, int, int] | None:
    for cf in cfs:
        for sv in range(m + 1):
            for bv in range(m + 1):
                for v in range(1, m):
                    total = sv + bv + v
                    if least <= total <= m and sv * n + v * cf == tr * total:
                        return sv, bv, v, cf
    return None
