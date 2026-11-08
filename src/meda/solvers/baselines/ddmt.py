"""MTC and DDMT: multi-target 1:1 dilution along one de Bruijn path, with demand.

- MTC (Mitra, Roy, Bhattacharjee, Chakrabarty & Bhattacharya, "On-Chip Sample Preparation for
  Multiple Targets Using Digital Microfluidics", IEEE TCAD 33(8), 2014): every CF x / 2^n is a
  node of the dilution graph D(2, n); mixing the current droplet with buffer (0) or sample (1)
  moves s_{n-1}..s_0 to b s_{n-1}..s_1. One path from CF 0 through all targets, the shortest
  (an ATSP over the targets, edge weights the de Bruijn distances), is one chain of 1:1
  mix-splits on one mixer. Example (Figs. 5-6): {5, 11, 14}/16 takes 6 mix-split steps, 4
  sample and 3 buffer droplets.
- DDMT (Poddar, Wille, Rahaman & Bhattacharya, "Demand-Driven Multi-Target Sample Preparation on
  Resource-Constrained Digital Microfluidic Biochips", ACM TODAES 27(1), 2021, Algorithm 1): the
  MTC sequence with each step's droplet count D_i = (its targets' demand) + ceil(D_{i+1} / 2),
  last step first; a step of D_i droplets is ceil(D_i / 2) mix-splits. Example (Fig. 4): {3, 5}
  with demands {4, 8} at n 3 is the sequence {4, 6, 3, 5} with D = {2, 4, 8, 8}.

Our reading of what the papers leave open (`notes/lit_bench/selection.md`):
- The ATSP is solved exactly (Held-Karp) for up to `EXACT_TARGETS` targets, the paper's
  Lin-Kernighan heuristic being at most as good; above that, nearest neighbour (ties: lower CF).
- The paper's CFs are rounded to n bits; a long path's droplet is within 2^-n of its label,
  not equal to it. Our engine is exact: each target is delivered from the real droplet, and a
  droplet outside the target's tolerance, or with a denominator above the chip's limit, makes
  the problem not applicable. n is the targets' own (an exact target needs a dyadic CF); a
  tolerant target is first moved to the nearest x / 2^n within half its tolerance, n the
  smallest accuracy at which every target fits (the paper takes n as given).
- On MEDA a step's mix-splits run as v:v mixes of up to M // (2 v) pairs each (v the targets'
  droplet volume, one for all targets). DDMT's single-mixer schedule with k storage cells
  (SMkS, Algorithms 2-3) is not modelled: the plan has the same mixes, the chip's storage limit
  applies (`emit`).
- MTC is DDMT at demand 1; `ddmt` takes any demand.
"""

from __future__ import annotations

from math import ceil

from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, sample_and_buffer

EXACT_TARGETS = 14


def distance(a: int, b: int, n: int) -> int:
    """Least steps from CF a / 2^n to b / 2^n in D(2, n) (the paper's Algorithm 1)."""
    for k in range(n + 1):
        if b % (1 << (n - k)) == a >> k:
            return k
    return n


def bits_to(a: int, b: int, n: int) -> list[int]:
    """The edge labels (0 buffer, 1 sample) of the shortest path from a to b (Algorithm 2)."""
    k = distance(a, b, n)
    return [(b >> (n - k + m)) & 1 for m in range(k)]


def tour(xs: list[int], n: int) -> list[int]:
    """The order of the targets on the shortest path from CF 0 (ATSP)."""
    m = len(xs)
    if m <= EXACT_TARGETS:
        best: dict[tuple[int, int], tuple[int, int]] = {}
        for j in range(m):
            best[(1 << j, j)] = (distance(0, xs[j], n), -1)
        for mask in range(1, 1 << m):
            for j in range(m):
                if (mask, j) not in best:
                    continue
                cost = best[(mask, j)][0]
                for k in range(m):
                    if mask >> k & 1:
                        continue
                    key = (mask | 1 << k, k)
                    c = cost + distance(xs[j], xs[k], n)
                    if key not in best or c < best[key][0]:
                        best[key] = (c, j)
        full = (1 << m) - 1
        end = min(range(m), key=lambda j: (best[(full, j)][0], j))
        order, mask = [], full
        while end != -1:
            order.append(end)
            end, mask = best[(mask, end)][1], mask & ~(1 << end)
        return [xs[j] for j in reversed(order)]
    left, at, out = sorted(xs), 0, []
    while left:
        nxt = min(left, key=lambda x: (distance(at, x, n), x))
        out.append(nxt)
        left.remove(nxt)
        at = nxt
    return out


def sequence(xs: list[int], n: int) -> list[tuple[int, int]]:
    """MTC's mix-split sequence: (label x of the droplet made, fluid bit mixed in)."""
    seq: list[tuple[int, int]] = []
    at = 0
    for x in tour(sorted(set(xs)), n):
        for b in bits_to(at, x, n):
            at = (b << (n - 1)) | (at >> 1)
            seq.append((at, b))
    return seq


def demands(seq: list[tuple[int, int]], wanted: dict[int, int]) -> list[int]:
    """DDMT's Algorithm 1: droplets each step makes (a target is taken at its last step)."""
    last = {x: i for i, (x, _) in enumerate(seq)}
    d = [0] * len(seq)
    for x, k in wanted.items():
        d[last[x]] += k
    for i in range(len(seq) - 2, -1, -1):
        d[i] += ceil(d[i + 1] / 2)
    return d


def ddmt(instance: Instance) -> PaperGraph:
    s, b = sample_and_buffer(instance)
    targets = instance.targets
    volumes = {t.droplet_volume for t in targets}
    if len(volumes) != 1:
        raise NotApplicable("targets of different droplet volumes")
    v = volumes.pop()
    pairs = instance.chip.max_mix_volume // (2 * v)
    if pairs < 1:
        raise NotApplicable(f"a {v}:{v} mix exceeds M")
    n, xs = _labels(instance, s)
    seq = sequence(xs, n)
    wanted: dict[int, int] = {}
    for x, t in zip(xs, targets, strict=True):
        wanted[x] = wanted.get(x, 0) + t.demand
    need = demands(seq, wanted)
    nodes: list[tuple[In, ...]] = []
    pool: list[list[int]] = []  # [node, droplets left] of the previous step
    last = {x: i for i, (x, _) in enumerate(seq)}
    out: dict[int, list[int]] = {}
    for i, (x, bit) in enumerate(seq):
        mixes = ceil(need[i] / 2)
        sizes = [pairs] * (mixes // pairs) + ([mixes % pairs] if mixes % pairs else [])
        here: list[list[int]] = []
        for a in sizes:
            ins = [In(a * v, fluid=s if bit else b)]
            if i == 0:
                ins.append(In(a * v, fluid=b if bit else s))
            else:
                left = a
                for cell in pool:
                    use = min(left, cell[1])
                    if use:
                        ins.append(In(use * v, node=cell[0]))
                        cell[1] -= use
                        left -= use
                if left:
                    raise AssertionError("a step takes more droplets than the last one made")
            nodes.append(tuple(ins))
            here.append([len(nodes) - 1, 2 * a])
        if i == last.get(x, -1) and x in wanted:
            take = wanted[x]
            out[x] = []
            for cell in reversed(here):  # deliveries from the step's last mixes
                while take and cell[1]:
                    out[x].append(cell[0])
                    cell[1] -= 1
                    take -= 1
        pool = here
    roots: list[int] = []
    for x, t in zip(xs, targets, strict=True):
        roots += [out[x].pop(0) for _ in range(t.demand)]
    return PaperGraph(instance.num_fluids, tuple(nodes), tuple(roots))


def _labels(instance: Instance, s: int) -> tuple[int, list[int]]:
    """(n, numerators x) of the targets' CFs at accuracy n."""
    cfs = [t.composition.share(s) for t in instance.targets]
    exact = [t.tolerance.num == 0 for t in instance.targets]
    if all(exact):
        if any(c.den & (c.den - 1) for c in cfs):
            raise NotApplicable("a target's CF is not dyadic")
        n = max(c.den.bit_length() - 1 for c in cfs)
        return n, [c.num << (n - (c.den.bit_length() - 1)) for c in cfs]
    top = instance.chip.max_denominator.bit_length() - 1
    for n in range(1, top + 1):
        xs = []
        for c, t in zip(cfs, instance.targets, strict=True):
            x = (2 * c.num * (1 << n) + c.den) // (2 * c.den)  # nearest, ties up
            diff = Rational.of(x, 1 << n) - c
            if max(diff, -diff) * 2 > t.tolerance:
                break
            xs.append(x)
        else:
            break
    else:
        raise NotApplicable(
            f"no accuracy up to 2^{top} puts every target within half its tolerance"
        )
    if any(x in (0, 1 << n) for x in xs):
        raise NotApplicable("a target is a pure fluid")
    return n, xs
