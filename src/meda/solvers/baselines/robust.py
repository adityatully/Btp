"""Robust split-less multi-target dilution on MEDA (Poddar, Banerjee, Wille & Bhattacharya,
"Robust Multi-Target Sample Preparation on MEDA Biochips Obviating Waste Production", ACM TODAES
26(1), 2020, Sections 4-6; SPEC 10.4).

Every target droplet is one mix, with no split, of aliquots of exponential concentration factors
(ECFs) c_i = 2^i / 2^n, i = 0..n. The paper's ILP (Eqs. 1-4) chooses aliquot counts w_i that
minimise the target droplet's volume k = sum w_i with the CF within tolerance. The ECFs come from
split-less exponential dilution chains (Algorithm 1): ECF 2^-x is a 1:1 mix of 2^-(x-1) and
buffer whose volume is exactly what is used, so nothing is split. Example 5: 71/128 at n 7 is
{2, 8, 32, 128}/128 in 1, 1, 1, 3 aliquots, k 6.

Our reading of what the paper leaves open (`notes/lit_bench/selection.md`):
- One aliquot is one volume unit of the chip (the paper's 1/16 X); the target droplet's k is at
  least its droplets' volume, and every mix is at most the chip's M units (the paper has no M:
  its chains reach 4X droplets, so on our chips deep ECFs often do not fit, and such a problem is
  not applicable).
- The ILP needs no solver: k aliquots of powers of two up to 2^n sum to T = C k 2^n exactly when
  q + popcount(r) <= k <= T, with T = q 2^n + r. The least k is taken; the paper's tie (which w)
  is left to its solver: we start from that binary form and split the smallest aliquot above one
  until there are k. An exact target needs a dyadic CF (n = its denominator's log2); a tolerant
  one is moved to the CF at 2^-n, n = log2 of the chip's denominator limit, of least k, then
  least error (the paper's tolerance is half a step at its accuracy level; ours is the target's).
- A target is delivered from ceil(d v / M) mixes when its demand d droplets of v units do not fit
  one mix; a target equal to an ECF is taken from its chain directly.
- Chains are built for what all targets need together (the paper stores ECFs in reservoirs and
  shares them across targets); a chain mix of 2a units takes a units of the previous ECF and a
  of buffer, a <= M // 2, and the leftover of an odd need is waste in our accounting (the paper
  keeps it in a reservoir).
"""

from __future__ import annotations

from math import ceil

from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, sample_and_buffer


def aliquots(t: int, n: int, k: int) -> list[int] | None:
    """Exponents j (aliquot of ECF 2^j / 2^n) of k aliquots summing to t, or None."""
    q, rest = divmod(t, 2**n)
    parts = [n] * q + [j for j in range(n) if rest >> j & 1]
    if not len(parts) <= k <= t:
        return None
    while len(parts) < k:
        j = min(p for p in parts if p > 0)
        parts.remove(j)
        parts += [j - 1, j - 1]
    return sorted(parts, reverse=True)


def least(num: int, n: int, low: int, high: int) -> tuple[int, list[int]] | None:
    """Least k in [low, high] with k aliquots making num / 2^n, and the aliquots."""
    for k in range(max(low, 1), high + 1):
        parts = aliquots(num * k, n, k)
        if parts is not None:
            return k, parts
    return None


def robust(instance: Instance) -> PaperGraph:
    s, b = sample_and_buffer(instance)
    chip = instance.chip
    big, half = chip.max_mix_volume, chip.max_mix_volume // 2
    # per target: (n, list of (droplets, k, aliquot exponents)) for its root mixes
    plans: list[tuple[int, list[tuple[int, int, list[int]]]]] = []
    for t in instance.targets:
        per_mix = big // t.droplet_volume
        if per_mix < 1:
            raise NotApplicable(f"target {t.id}: a droplet of {t.droplet_volume} units exceeds M")
        groups = [per_mix] * (t.demand // per_mix) + (
            [t.demand % per_mix] if t.demand % per_mix else []
        )
        best: tuple[tuple[int, Rational], int, int, list[tuple[int, int, list[int]]]] | None = None
        for n, num in _cfs(instance, t.id, s):
            roots: list[tuple[int, int, list[int]]] = []
            for g in groups:
                found = least(num, n, g * t.droplet_volume, big)
                if found is None:
                    break
                roots.append((g, *found))
            else:
                diff = Rational.of(num, 2**n) - t.composition.share(s)
                err = max(diff, -diff)
                key = (sum(k for _, k, _ in roots), err)
                if best is None or key < best[0]:
                    best = (key, n, num, roots)
        if best is None:
            raise NotApplicable(
                f"target {t.id}: no split-less mix of at most {big} ECF aliquots makes it"
            )
        plans.append((best[1], best[3]))
    n = max(p[0] for p in plans)
    # need[x]: units of ECF 2^-x the targets take (x = n - j after moving every plan to depth n)
    need = [0] * (n + 1)
    for depth, roots in plans:
        for _, _k, parts in roots:
            if len(set(parts)) == 1:
                continue  # delivered from the chain itself, counted below
            for j in parts:
                need[depth - j] += 1
    direct = [0] * (n + 1)
    for depth, roots in plans:
        for _g, k, parts in roots:
            if len(set(parts)) == 1:
                x = depth - parts[0]
                if x == 0:
                    raise NotApplicable("a target is the pure sample")
                direct[x] += k
    nodes: list[tuple[In, ...]] = []
    chain: list[list[tuple[int, int]]] = [[] for _ in range(n + 1)]  # level -> (node, units)
    sizes: list[list[int]] = [[] for _ in range(n + 1)]
    feed = 0
    for x in range(n, 0, -1):
        total = need[x] + direct[x] + feed
        if total == 0:
            continue
        a_total = ceil(total / 2)
        m = ceil(a_total / half) if half else 0
        if m == 0:
            raise NotApplicable("the chip cannot mix 1:1")
        sizes[x] = [a_total // m + (1 if i < a_total % m else 0) for i in range(m)]
        feed = a_total
    pool: dict[int, list[list[int]]] = {}  # level -> [[node, units left]]

    def take(x: int, units: int) -> list[In]:
        if x == 0:
            return [In(units, fluid=s)]
        out: list[In] = []
        for cell in pool[x]:
            if units == 0:
                break
            use = min(units, cell[1])
            if use:
                out.append(In(use, node=cell[0]))
                cell[1] -= use
                units -= use
        if units:
            raise AssertionError("chain smaller than its uses")
        return out

    for x in range(1, n + 1):
        pool[x] = []
        for a in sizes[x]:
            nodes.append((*take(x - 1, a), In(a, fluid=b)))
            pool[x].append([len(nodes) - 1, 2 * a])
            chain[x].append((len(nodes) - 1, 2 * a))
    roots_out: list[int] = []
    for (depth, roots), t in zip(plans, instance.targets, strict=True):
        for g, _k, parts in roots:
            if len(set(parts)) == 1:
                x = depth - parts[0]
                cell = next((c for c in pool[x] if c[1] >= g * t.droplet_volume), None)
                if cell is None:
                    raise NotApplicable(f"target {t.id}: no chain mix holds its droplets")
                cell[1] -= g * t.droplet_volume
                roots_out += [cell[0]] * g
                continue
            ins: list[In] = []
            for j in sorted(set(parts), reverse=True):
                ins += take(depth - j, parts.count(j))
            nodes.append(tuple(ins))
            roots_out += [len(nodes) - 1] * g
    return PaperGraph(instance.num_fluids, tuple(nodes), tuple(roots_out))


def _cfs(instance: Instance, target: int, s: int) -> list[tuple[int, int]]:
    """(n, numerator) of the dyadic CFs the target accepts."""
    t = instance.targets[target]
    c = t.composition.share(s)
    den = c.den
    if t.tolerance == Rational.of(0):
        if den & (den - 1):
            raise NotApplicable(f"target {target}: CF {c} is not dyadic")
        n = den.bit_length() - 1
        return [(n, c.num)]
    limit = instance.chip.max_denominator
    if limit & (limit - 1):
        raise NotApplicable("the chip's denominator limit is not a power of two")
    n = limit.bit_length() - 1
    lo, hi = c - t.tolerance, c + t.tolerance
    return [(n, m) for m in range(1, 2**n) if lo <= Rational.of(m, 2**n) <= hi]
