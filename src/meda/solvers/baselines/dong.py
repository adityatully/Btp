"""Dong's splitting-droplet sharing: dilution gradients from split sample droplets (Dong, Chen &
Chen, "Reactant and Waste Minimization during Sample Preparation on Micro-Electrode-Dot-Array
Digital Microfluidic Biochips using Splitting Trees", JETTA 40(1), 2024; SPEC 10.4).

A target CF x / 2^h is a droplet of 2^h unit cells. A sample droplet is split in halves, level
by level (a splitting tree), into pieces of 2^k cells; the pieces of x's 1 bits are merged with
2^h - x cells of buffer in one mix. Several targets take their pieces from shared trees, and the
pieces no target takes are kept for later targets (the waste-sharing stage). Fig. 10: 848 / 1024
is 512 + 256 + 64 + 16 from one sample droplet, four splits and one mix. Fig. 8: the gradient
{64, 160, 208, 232, 244, 248, 252} / 1024 takes six 128s, six 64s, five 32s, four 16s, three 8s
and two 4s.

Our reading (`notes/lit_bench/selection.md`):
- The paper's reading is fixed by its own numbers: Fig. 8's piece counts are the 1 bits of the
  targets, and Table 3's single-target counts (one sample droplet, 9 down to 1 operations and
  waste pieces for x = 2 to 512) are the splits of one sample droplet. Its "CF" of a piece is
  that piece's share of one droplet's volume; buffer is free.
- Our engine dispenses any whole number of units from a reservoir, so the splitting tree costs
  nothing and leaves no pieces: what remains of the method is one sample-buffer mix of 2^h units
  per target droplet, x units of sample. The paper's sample count (whole sample droplets) and
  waste (unused pieces) have no counterpart; ours counts units.
- The CF must be x / 2^h with 2^h <= M (the paper's targets are h-bit fractions; a droplet of
  2^h cells must fit one mix). A tolerant target is moved to the x / 2^h nearest it, the smallest
  h whose nearest point is within tolerance. Demand d of a v-unit droplet is served by mixes of
  2^h ceil(q v / 2^h) <= M units holding q droplets each, as many as fit.
"""

from __future__ import annotations

from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph, sample_and_buffer


def dong(instance: Instance) -> PaperGraph:
    s, b = sample_and_buffer(instance)
    m = instance.chip.max_mix_volume
    nodes: list[tuple[In, ...]] = []
    roots: list[int] = []
    for t in instance.targets:
        x, den = _dyadic(instance, t.id, s)
        if x in (0, den):
            raise NotApplicable(f"target {t.id} is a pure fluid")
        v = t.droplet_volume
        per = max((q for q in range(1, m // v + 1) if _size(q, v, den) <= m), default=0)
        if per == 0:
            raise NotApplicable(f"target {t.id}: a {den}-unit mix for a {v}-unit droplet exceeds M")
        left = t.demand
        while left:
            q = min(per, left)
            size = _size(q, v, den)
            nodes.append((In(x * size // den, fluid=s), In((den - x) * size // den, fluid=b)))
            roots += [len(nodes) - 1] * q
            left -= q
    return PaperGraph(instance.num_fluids, tuple(nodes), tuple(roots))


def _size(q: int, v: int, den: int) -> int:
    return den * -(-q * v // den)


def _dyadic(instance: Instance, target: int, s: int) -> tuple[int, int]:
    """(x, 2^h) of the target's CF; the least h <= log2 M within tolerance."""
    t = instance.targets[target]
    c = t.composition.share(s)
    top = instance.chip.max_mix_volume.bit_length() - 1
    for h in range(top + 1):
        den = 1 << h
        x = (2 * c.num * den + c.den) // (2 * c.den)  # nearest, ties up
        diff = Rational.of(x, den) - c
        if max(diff, -diff) <= t.tolerance:
            return x, den
    raise NotApplicable(f"target {target}: no x / 2^h within tolerance with 2^h <= M")
