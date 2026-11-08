"""SIMOP: simulation-guided search for the best (1:1) mix-split chain of one dilution target
(Bera, Majumder, Das, Mukherjee, Aryani & Bhattacharya, "SIMOP: A SIMulation-Guided OPtimization
Mechanism for Sample Preparation with Digital Microfluidic Biochip", SN Computer Science 2:84,
2021, Part I; SPEC 10.4).

A chain starts by mixing sample and buffer; each later step mixes the droplet just made with a
droplet of a different CF: sample, buffer, or the spare second droplet of an earlier step (each
mix-split makes two). SIMOP enumerates every such chain up to the accuracy limit x and keeps,
for each CF, the best chain under a user-ordered list of criteria: waste > steps > sample >
buffer, sample > waste > steps > buffer, or buffer > waste > steps > sample. Waste is every
droplet left besides the two target droplets. Table 5: 313/1024 takes 2 sample, 3 buffer and
3 waste ordered by buffer, 1 sample, 4 buffer, 3 waste ordered by sample. Table 6: averages over
the 1023 CFs at x = 10 are waste 2.59, sample 2.30, buffer 2.30, steps 9.01.

Our reading (`notes/lit_bench/selection.md`):
- The paper's look-up table over all CFs is replaced by a search for the one CF asked: a
  depth-first search over the same chains with the same orders, pruned by the range a chain can
  still reach and by the criteria's lower bounds. A CF x / 2^h takes exactly h steps: a chain's
  k-th droplet has denominator 2^k, as every partner's is smaller.
- On MEDA a droplet is the target's droplet volume v and a mix is v:v (2v <= M). Demand d is the
  chain run ceil(d / 2) times (SIMOP makes one target, two droplets). The CF must be x / 2^h
  with 2^h at most the chip's denominator limit and h <= `MAX_BITS`; a tolerant target is moved
  to the x / 2^h nearest it at the smallest h within tolerance. Part II (split errors) is not
  modelled: our engine is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, sample_and_buffer

ORDERS = {
    "waste": ("waste", "steps", "sample", "buffer"),
    "sample": ("sample", "waste", "steps", "buffer"),
    "buffer": ("buffer", "waste", "steps", "sample"),
}
MAX_BITS = 10  # the paper's own accuracy; 11 bits already takes over 9 s a CF
BUFFER, SAMPLE = -1, -2  # partner codes; a spare is the index of its step


@dataclass
class Search:
    t: int  # target numerator over D = 2^h
    h: int
    order: tuple[str, ...]
    best: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    seen: dict[tuple[int, int, tuple[int, ...]], list[tuple[int, int]]] = field(
        default_factory=dict
    )

    def key(self, sample: int, buffer: int) -> tuple[int, ...]:
        v = {"waste": sample + buffer - 2, "steps": self.h, "sample": sample, "buffer": buffer}
        return tuple(v[c] for c in self.order)

    def run(self) -> tuple[int, ...]:
        d = 1 << self.h
        # step 0 mixes sample and buffer: CF D / 2, one droplet goes on, one is spare
        self._go(1, d // 2, [d // 2], [False], 1, 1, [])
        if self.best is None:
            raise AssertionError("bit scanning is one of SIMOP's chains")
        return self.best[1]

    def _go(
        self,
        k: int,
        c: int,
        values: list[int],
        used: list[bool],
        s: int,
        b: int,
        partners: list[int],
    ) -> None:
        d, r = 1 << self.h, self.h - k
        if r == 0:
            if c == self.t:
                key = self.key(s, b)
                if self.best is None or key < self.best[0]:
                    self.best = (key, tuple(partners))
            return
        lo = c  # the final CF times 2^r lies in [c, c + D (2^r - 1)]
        if not lo <= self.t << r <= lo + d * ((1 << r) - 1):
            return
        if self.best is not None and self.key(s, b) >= self.best[0]:
            return  # every key part only grows (steps is fixed)
        spares = tuple(sorted(v for v, u in zip(values, used, strict=True) if not u))
        state = (k, c, spares)
        for s0, b0 in self.seen.get(state, []):
            if s0 <= s and b0 <= b:
                return
        self.seen.setdefault(state, []).append((s, b))
        options: list[tuple[int, int]] = [(0, BUFFER), (d, SAMPLE)]
        taken: set[int] = set()
        for j, (v, u) in enumerate(zip(values, used, strict=True)):
            if not u and v not in taken:
                taken.add(v)
                options.append((v, j))
        for p, code in options:
            if p == c:
                continue
            new = (c + p) // 2
            if code >= 0:
                used[code] = True
            self._go(
                k + 1,
                new,
                [*values, new],
                [*used, False],
                s + (code == SAMPLE),
                b + (code == BUFFER),
                [*partners, code],
            )
            if code >= 0:
                used[code] = False


def chain(x: int, h: int, criterion: str = "waste") -> tuple[int, ...]:
    """SIMOP's partners for CF x / 2^h, step 1 on: BUFFER, SAMPLE, or the step whose spare
    droplet is taken (step 0 mixes sample and buffer)."""
    if h > MAX_BITS:
        raise NotApplicable(f"2^{h} is above SIMOP's search limit 2^{MAX_BITS}")
    return Search(x, h, ORDERS[criterion]).run()


def counts(partners: tuple[int, ...]) -> tuple[int, int, int, int]:
    """(steps, sample, buffer, waste) in droplets, as the paper counts them."""
    s = 1 + partners.count(SAMPLE)
    b = 1 + partners.count(BUFFER)
    return len(partners) + 1, s, b, s + b - 2


def simop(instance: Instance, criterion: str = "waste") -> PaperGraph:
    if len(instance.targets) != 1:
        raise NotApplicable("SIMOP makes one target")
    s, b = sample_and_buffer(instance)
    t = instance.targets[0]
    v = t.droplet_volume
    if 2 * v > instance.chip.max_mix_volume:
        raise NotApplicable(f"a {v}:{v} mix exceeds M")
    x, h = _dyadic(instance, s)
    if h == 0:
        raise NotApplicable("the target is a pure fluid")
    partners = chain(x, h, criterion)
    nodes: list[tuple[In, ...]] = []
    roots: list[int] = []
    for copy in range(ceil(t.demand / 2)):
        base = len(nodes)
        nodes.append((In(v, fluid=s), In(v, fluid=b)))
        for code in partners:
            other = {SAMPLE: In(v, fluid=s), BUFFER: In(v, fluid=b)}.get(code)
            nodes.append((In(v, node=len(nodes) - 1), other or In(v, node=base + code)))
        roots += [len(nodes) - 1] * min(2, t.demand - 2 * copy)
    return PaperGraph(instance.num_fluids, tuple(nodes), tuple(roots))


def _dyadic(instance: Instance, s: int) -> tuple[int, int]:
    t = instance.targets[0]
    c = t.composition.share(s)
    top = instance.chip.max_denominator.bit_length() - 1
    for h in range(top + 1):
        den = 1 << h
        x = (2 * c.num * den + c.den) // (2 * c.den)  # nearest, ties up
        diff = Rational.of(x, den) - c
        if max(diff, -diff) <= t.tolerance:
            return x, h
    raise NotApplicable("no x / 2^h within tolerance under the chip's denominator limit")
