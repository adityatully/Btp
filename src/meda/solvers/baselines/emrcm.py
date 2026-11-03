"""e-MRCM: MRCM for several targets, sharing operations across their trees (Liang et al.,
"Multi-Target Sample Preparation Using MEDA Biochips", IEEE TCAD 39(10), 2020, Section V;
SPEC 10.4).

The paper grows the targets' MRCM trees together: the open leaf with the largest concentration
sum goes first, and a half-grown tree that contains a node another tree also needs is kept over
those that do not; at the end, common operations are shared. Fig. 11: for <25, 31, 44> and the
second target, W = <2, 1, 3>, separate MRCM trees cost 46 and sharing <0, 3, 22> brings it to 28
(Fig. 11(c): 4 mixes, R1 4, R2 2, R3 6, waste 4).

Our reading of what the paper leaves open (`notes/lit_bench/objectives.md`):
- Targets are taken one at a time, largest reduced denominator (the paper's concentration
  sum) first, ties in target order. Each is an MRCM search (`mrcm._Search`) in which a child
  equal to a droplet an earlier target's tree made, with enough of it left over, costs nothing
  and needs no mix: that is "keep the tree with a sharable node", and sharing it at once is
  "share common operations". A target equal to such a droplet is taken from it directly.
- A shared droplet is taken from at most once per target tree; if the cheapest tree would take
  more than is left, that droplet is not offered and the target is searched again.
- One droplet per target (demand 1), every target built exactly (MRCM's rule); otherwise not
  applicable.
- The search is MRCM's (`mrcm._Search`). The paper's narrower one (Eq. 11, the first term of
  Eq. 10 set to 1) is optional (`narrow=True`), read as: a mix of T units for a node of
  denominator D needs T | D, and its intermediate children's denominators divide D / T
  together. Every tree the paper draws (Figs. 3 and 11) meets it, but on random targets at
  1/512 it leaves most 4-reactant targets with no tree, which the paper never reports; it is not
  the paper's search (paper log L-7, Q-117).
- The paper's second target is printed as <3, 6, 44>, which no mix of its Fig. 11 tree makes;
  that tree (3 units of R1 with 2 of <0, 3, 22>) makes <75, 6, 44>/125, which we use.
"""

from __future__ import annotations

from dataclasses import dataclass

from meda.core.composition import Composition, Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.solvers.baselines.mrcm import _Search, _Tree, smooth
from meda.solvers.baselines.paper_graph import In, NotApplicable, PaperGraph


@dataclass(frozen=True, slots=True)
class _Shared(_Tree):
    """A child taken from a droplet an earlier target's tree made: no reactant, no mix."""

    node: int = -1


class _SharedSearch(_Search):
    def __init__(
        self,
        w: tuple[Rational, ...],
        lo: int,
        m: int,
        spare: dict[Composition, tuple[int, int]],
        narrow: bool,
    ) -> None:
        super().__init__(w, lo, m, narrow)
        self.spare = spare  # composition -> (node, units left over)

    def grow(self, model, v: int) -> _Tree | None:
        t, pure, inter = model
        cost = sum((w * p for w, p in zip(self.w, pure, strict=True)), Rational.of(0))
        mixes, waste = 1, t - v
        children: list[tuple[int, _Tree]] = []
        for units, comp in inter:
            have = self.spare.get(comp)
            if have is not None and have[1] >= units:
                children.append((units, _Shared((Rational.of(0), 0, 0), (), (), 0, have[0])))
                waste -= units  # what the earlier tree would have wasted
                continue
            sub = self.best(comp, units)
            if sub is None:
                return None
            children.append((units, sub))
            cost = cost + sub.key[0]
            mixes += sub.key[1]
            waste += sub.key[2]
        return _Tree((cost, mixes, waste), tuple(pure), tuple(children), t)


def emrcm(instance: Instance, weights: CostWeights) -> PaperGraph:
    targets, chip = instance.targets, instance.chip
    if any(t.demand != 1 for t in targets):
        raise NotApplicable("e-MRCM makes one droplet of each target (demand 1)")
    m = chip.max_mix_volume
    for t in targets:
        if not smooth(t.composition.den, m):
            raise NotApplicable(
                f"{t.composition}: denominator has a prime factor above M = {m} (e-MRCM builds "
                f"every target exactly)"
            )
    w = tuple(weights.fluid_weight(f.kind) for f in instance.fluids)
    return emrcm_graph(
        [(t.composition, t.droplet_volume) for t in targets],
        w,
        max(chip.min_mix_volume, 2),
        m,
        instance.num_fluids,
    )


def emrcm_graph(
    targets: list[tuple[Composition, int]],
    w: tuple[Rational, ...],
    lo: int,
    m: int,
    num_fluids: int,
    narrow: bool = False,
) -> PaperGraph:
    """The shared graph for `targets` (composition, droplet volume), roots in their order;
    `narrow` is the paper's Eq. 11 search, False the full MRCM search."""
    nodes: list[tuple[In, ...]] = []
    left: list[int] = []  # units of each node no one has taken yet
    spare: dict[Composition, tuple[int, int]] = {}
    roots = [-1] * len(targets)
    order = sorted(range(len(targets)), key=lambda i: (-targets[i][0].den, i))
    for i in order:
        comp, vol = targets[i]
        have = spare.get(comp)
        if have is not None and have[1] >= vol:
            roots[i] = have[0]
            _take(have[0], vol, left, nodes, spare)
            continue
        offered = dict(spare)
        while True:
            tree = _SharedSearch(w, lo, m, offered, narrow).best(comp, vol)
            if tree is None:
                raise NotApplicable(f"e-MRCM finds no tree for {comp} at M = {m}")
            over = _overdrawn(tree, spare)
            if over is None:
                break
            del offered[over]
        roots[i] = _emit(tree, nodes, left, spare)
        _take(roots[i], vol, left, nodes, spare)
    return PaperGraph(num_fluids, tuple(nodes), tuple(roots))


def _overdrawn(tree: _Tree, spare: dict[Composition, tuple[int, int]]) -> Composition | None:
    """A shared droplet the tree takes more of than is left, if any."""
    asked: dict[int, int] = {}
    stack = [tree]
    while stack:
        t = stack.pop()
        for units, child in t.children:
            if isinstance(child, _Shared):
                asked[child.node] = asked.get(child.node, 0) + units
            else:
                stack.append(child)
    for comp, (node, units) in spare.items():
        if asked.get(node, 0) > units:
            return comp
    return None


def _emit(
    tree: _Tree,
    nodes: list[tuple[In, ...]],
    left: list[int],
    spare: dict[Composition, tuple[int, int]],
) -> int:
    ins = [In(u, fluid=f) for f, u in enumerate(tree.pure) if u]
    for units, child in tree.children:
        src = child.node if isinstance(child, _Shared) else _emit(child, nodes, left, spare)
        ins.append(In(units, node=src))
        _take(src, units, left, nodes, spare)
    nodes.append(tuple(ins))
    left.append(tree.volume)
    num = len(nodes) - 1
    comp = PaperGraph(len(tree.pure), tuple(nodes)).compositions()[num]
    spare.setdefault(comp, (num, tree.volume))
    return num


def _take(
    node: int,
    units: int,
    left: list[int],
    nodes: list[tuple[In, ...]],
    spare: dict[Composition, tuple[int, int]],
) -> None:
    left[node] -= units
    for comp, (n, _) in list(spare.items()):
        if n == node:
            if left[node] > 0:
                spare[comp] = (node, left[node])
            else:
                del spare[comp]
